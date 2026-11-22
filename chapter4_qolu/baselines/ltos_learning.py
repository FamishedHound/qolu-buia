"""The bi-level LToS update (Algorithm 1 of Yi et al., condensed in the note).

Split out of :mod:`chapter4_qolu.baselines.ltos` to keep both files small.  Two
updates run here, in the order the published sketch gives them:

1. **Low level** - DGN Q-learning on the shaped reward ``r_i^w`` with a soft
   target network;
2. **High level** - the first-order bi-level step.  The DGN Q-network doubles as
   the DDPG critic (the identity ``q^{phi_i}(o_i, w_i^out) = v^{pi_i}(o_i ;
   w_i^in)``), so the gradient ``g_i^in`` is taken w.r.t. ``w_i^in`` at the
   argmax low-level action, exchanged into ``g_i^out``, and pushed through the
   actor.  Second-order terms are dropped, as in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from chapter4_qolu.baselines.config import LToSConfig
from chapter4_qolu.baselines.districts import N_SLOTS, REVERSE_SLOT, DistrictLattice
from chapter4_qolu.baselines.ltos_networks import DGNQNetwork, SharingActor
from chapter4_qolu.baselines.ltos_replay import LToSTransitionBatch

__all__ = [
    "GraphTensors",
    "exchange_gradients",
    "soft_update",
    "update_high_level",
    "update_low_level",
]


@dataclass(frozen=True)
class GraphTensors:
    """The fixed lattice adjacency, as tensors the networks can consume."""

    neighbour_index: torch.Tensor
    neighbour_mask: torch.Tensor

    @classmethod
    def for_lattice(
        cls, lattice: DistrictLattice, device: torch.device
    ) -> GraphTensors:
        return cls(
            neighbour_index=torch.as_tensor(
                lattice.neighbour_index, dtype=torch.long, device=device
            ),
            neighbour_mask=torch.as_tensor(
                lattice.neighbour_mask, dtype=torch.float32, device=device
            ),
        )


def _float(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray(array, dtype=np.float32), dtype=torch.float32, device=device
    )


def _long(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=torch.long, device=device)


def update_low_level(
    q_network: DGNQNetwork,
    q_target: DGNQNetwork,
    optimizer: torch.optim.Optimizer,
    batch: LToSTransitionBatch,
    graph: GraphTensors,
    config: LToSConfig,
    device: torch.device,
) -> float:
    """One DGN Q-learning step on the shaped reward.

    Design note: every candidate parcel inside a district is one decision *by
    that district*, so each of them is a transition earning the district's
    shaped reward ``r_i^w``.  A district with several candidates in the buffer
    therefore contributes several transitions, which is the natural reading of
    "the action ``a_i`` is convert / skip for each candidate parcel inside the
    district".
    """
    districts = _long(batch.parcel_districts, device)
    next_districts = _long(batch.next_parcel_districts, device)
    actions = _long(batch.actions, device)
    rewards = _float(batch.rewards, device)
    dones = _float(batch.dones, device).view(-1, 1)

    q_values = q_network(
        _float(batch.node_features, device),
        _float(batch.sharing_in, device),
        _float(batch.parcel_features, device),
        districts,
        graph.neighbour_index,
        graph.neighbour_mask,
    )
    taken = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_q = q_target(
            _float(batch.next_node_features, device),
            _float(batch.next_sharing_in, device),
            _float(batch.next_parcel_features, device),
            next_districts,
            graph.neighbour_index,
            graph.neighbour_mask,
        ).max(dim=-1)[0]
        target = rewards.gather(1, districts) + config.gamma * (1.0 - dones) * next_q

    loss = torch.nn.functional.mse_loss(taken, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if config.max_grad_norm is not None:
        nn.utils.clip_grad_norm_(q_network.parameters(), config.max_grad_norm)
    optimizer.step()
    return float(loss.detach().cpu())


def update_high_level(
    q_network: DGNQNetwork,
    actor: SharingActor,
    optimizer: torch.optim.Optimizer,
    batch: LToSTransitionBatch,
    graph: GraphTensors,
    lattice: DistrictLattice,
    config: LToSConfig,
    device: torch.device,
) -> None:
    """``g_i^in -> exchange -> g_i^out -> theta_i`` (the first-order bi-level step)."""
    sharing_in = _float(batch.sharing_in, device).requires_grad_(True)
    districts = _long(batch.parcel_districts, device)
    q_values = q_network(
        _float(batch.node_features, device),
        sharing_in,
        _float(batch.parcel_features, device),
        districts,
        graph.neighbour_index,
        graph.neighbour_mask,
    )
    best = q_values.max(dim=-1)[0]
    assignment = torch.nn.functional.one_hot(
        districts, num_classes=lattice.n_districts
    ).to(best.dtype)
    totals = torch.einsum("bm,bmk->bk", best, assignment)
    counts = assignment.sum(dim=1).clamp(min=1.0)
    # Design note: a district's value is the *mean* over the candidates it was
    # given, so districts that happen to be sampled more often do not dominate
    # the high-level gradient.  Districts with no candidates contribute nothing.
    value = (totals / counts).sum()
    gradient = torch.autograd.grad(value, sharing_in)[0].detach().cpu().numpy()

    exchanged = _float(exchange_gradients(gradient, lattice), device)
    weights = actor(_float(batch.node_features, device), graph.neighbour_mask)
    loss = -(weights * exchanged).sum(dim=-1).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if config.max_grad_norm is not None:
        nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
    optimizer.step()


def exchange_gradients(gradient: np.ndarray, lattice: DistrictLattice) -> np.ndarray:
    """Send each ``g_i^in`` slot back to the neighbour that supplied the weight."""
    exchanged = np.zeros_like(gradient)
    for slot in range(N_SLOTS):
        for district in range(lattice.n_districts):
            if lattice.neighbour_mask[district, slot] == 0.0:
                continue
            sender = int(lattice.neighbour_index[district, slot])
            exchanged[:, sender, REVERSE_SLOT[slot]] += gradient[:, district, slot]
    return exchanged


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging of the DGN target network (published tau = 0.1)."""
    with torch.no_grad():
        for target_parameter, parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)
