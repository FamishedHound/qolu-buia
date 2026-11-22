"""Bottom-up investor agent (BUIA) - Algorithm 2 of the paper.

Algorithm 2, per decision point:

    Z = NeuralNetwork(X; theta);  P = softmax(Z);  a ~ Categorical(P);
    execute a (convert selected parcels);  observe R_NN from the change in
    attractiveness.

The feature set ``X`` is the agent's local context - "land cover, neighbours'
attractiveness updates" - restricted to a 2 km radius, which the environment
serves through :meth:`UrbanLandUseEnv.buia_observations`.

Design note (training rule): the paper never states how BUIA is trained, so the
policy is updated with REINFORCE on its own reward, with an exponential
moving-average baseline.  See :class:`chapter4_qolu.config.BUIAConfig`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from chapter4_qolu.agent.buia_network import BUIANetwork, BUIANetworkConfig
from chapter4_qolu.agent.proposal_sampling import sample_without_replacement
from chapter4_qolu.config import BUIAConfig, EnvConfig
from chapter4_qolu.env.parcel_grid import ParcelIndex

__all__ = ["BUIAAgent", "BUIAConfig", "Proposal", "buia_proposal_limit"]


def buia_proposal_limit(config: EnvConfig) -> int:
    """How many parcels BUIA may put forward in one iteration.

    Design note: with ``max_conversions_per_iteration = None`` - the campaign's
    ``cap none`` - the investor may propose the *whole* decision buffer, since
    "no cap" means every approved candidate converts.  ``BUIAConfig`` requires a
    positive integer, so the limit cannot simply be left unset;
    ``samples_per_iteration`` is the largest number of distinct parcels the
    buffer can contain (Eq. 2 samples it with replacement, so usually fewer).

    Lives beside the agent so that every method with an investor - QOLU + BUIA
    and the ``no_regulator`` ablation - sizes it by one rule.
    """
    if config.max_conversions_per_iteration is not None:
        return int(config.max_conversions_per_iteration)
    return int(config.samples_per_iteration)


@dataclass(frozen=True)
class Proposal:
    """The parcels BUIA wants converted this iteration."""

    parcels: tuple[ParcelIndex, ...]
    indices: tuple[int, ...]
    log_probability: float


class BUIAAgent:
    """Neural-network investor proposing profitable local land conversions."""

    def __init__(self, config: BUIAConfig, seed: int | None = None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.network = BUIANetwork(
            BUIANetworkConfig(
                in_channels=config.observation_channels,
                window=config.observation_window,
                conv_filters=config.conv_filters,
                conv_kernel=config.conv_kernel,
                mlp_units=config.mlp_units,
            )
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate
        )
        self.baseline = 0.0
        self.updates = 0
        self._pending_log_probability: torch.Tensor | None = None
        self._pending_entropy: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # acting
    # ------------------------------------------------------------------
    def propose(
        self,
        candidates: Sequence[ParcelIndex],
        observations: np.ndarray,
        max_proposals: int | None = None,
    ) -> Proposal:
        """Sample up to ``max_proposals`` distinct parcels to convert."""
        candidates = tuple(candidates)
        tensor = self._validate(candidates, observations)

        logits = self.network(tensor)
        limit = self.config.max_proposals if max_proposals is None else max_proposals
        indices, log_probability = sample_without_replacement(logits, limit, self.rng)

        log_probabilities = torch.log_softmax(logits, dim=0)
        self._pending_log_probability = log_probability
        self._pending_entropy = -(log_probabilities.exp() * log_probabilities).sum()

        return Proposal(
            parcels=tuple(candidates[index] for index in indices),
            indices=tuple(indices),
            log_probability=float(log_probability.detach().cpu()),
        )

    @torch.no_grad()
    def proposal_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """``P = softmax(Z)`` over the current candidate parcels."""
        tensor = torch.as_tensor(
            np.asarray(observations, dtype=np.float32), device=self.device
        )
        return self.network.distribution(tensor).cpu().numpy()

    @torch.no_grad()
    def log_probability_of(
        self, indices: Sequence[int], observations: np.ndarray
    ) -> float:
        """Log-probability the current policy assigns to an ordered selection."""
        tensor = torch.as_tensor(
            np.asarray(observations, dtype=np.float32), device=self.device
        )
        _, log_probability = sample_without_replacement(
            self.network(tensor),
            len(tuple(indices)),
            self.rng,
            forced_indices=tuple(indices),
        )
        return float(log_probability.cpu())

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def update(self, reward: float) -> float:
        """One REINFORCE step on the last proposal with return ``R_NN``."""
        if self._pending_log_probability is None or self._pending_entropy is None:
            raise RuntimeError("call propose() before update()")

        advantage = float(reward) - self.baseline
        loss = -advantage * self._pending_log_probability
        if self.config.entropy_coefficient > 0.0:
            loss = loss - self.config.entropy_coefficient * self._pending_entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self.network.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()

        decay = self.config.baseline_decay
        self.baseline = decay * self.baseline + (1.0 - decay) * float(reward)
        self.updates += 1
        self._pending_log_probability = None
        self._pending_entropy = None
        return float(loss.detach().cpu())

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "baseline": self.baseline,
            "updates": self.updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.network.load_state_dict(state["network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.baseline = float(state.get("baseline", 0.0))
        self.updates = int(state.get("updates", 0))

    # ------------------------------------------------------------------
    def _validate(
        self, candidates: Sequence[ParcelIndex], observations: np.ndarray
    ) -> torch.Tensor:
        if not candidates:
            raise ValueError("at least one candidate parcel is required")
        array = np.asarray(observations, dtype=np.float32)
        if array.ndim != 4:
            raise ValueError(
                "observations must have shape (candidates, channels, window, window)"
            )
        if array.shape[0] != len(candidates):
            raise ValueError(
                f"got {array.shape[0]} observations for {len(candidates)} candidates"
            )
        window = self.config.observation_window
        if array.shape[1:] != (self.config.observation_channels, window, window):
            raise ValueError(
                f"expected observations of shape "
                f"({self.config.observation_channels}, {window}, {window}), "
                f"got {array.shape[1:]}"
            )
        return torch.as_tensor(array, device=self.device)
