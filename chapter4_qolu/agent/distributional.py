"""Categorical distributional RL primitives used by QOLU.

Naming / formulation caveat
---------------------------
The paper's title and prose describe *quantile regression* and quote the
quantile-regression loss

    L(phi) = 1/N sum_i sum_j rho_tau_j( y_ij - Q_psi(s_i, a_i, tau_j) ),

but Algorithm 1 and Table 1 describe something different: a fixed support of
``N`` equally spaced atoms on ``[V_min, V_max]`` (the "Atom support" paragraph
cites Bellemare et al. 2017), 51 atoms over [-200, 200], and "a gradient
descent step on the Kullback-Leibler divergence D_KL(Z || Z(s, a; theta))".
Those are the mechanics of *categorical* distributional RL (C51), not of
quantile regression (QR-DQN / IQN), which uses adjustable quantile locations
and an asymmetric pinball loss instead of a KL divergence.

This module implements what the pseudocode and hyper-parameter table actually
specify - the categorical formulation - and the names deliberately avoid
claiming either reading (hence ``categorical_kl_divergence`` rather than
``quantile_regression_loss``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

__all__ = [
    "AtomSupport",
    "categorical_kl_divergence",
    "combine_dueling_streams",
    "project_categorical_distribution",
]

Reduction = Literal["mean", "sum", "none"]


@dataclass(frozen=True)
class AtomSupport:
    """Fixed categorical support ``z_i = V_min + i * delta_z`` (Table 1 defaults)."""

    n_atoms: int = 51
    v_min: float = -200.0
    v_max: float = 200.0

    def __post_init__(self) -> None:
        if self.n_atoms < 2:
            raise ValueError("n_atoms must be at least two")
        if self.v_max <= self.v_min:
            raise ValueError("v_max must be greater than v_min")

    @property
    def delta_z(self) -> float:
        """Atom spacing ``(V_max - V_min) / (N - 1)``."""
        return (self.v_max - self.v_min) / (self.n_atoms - 1)

    def atoms(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """The canonical returns ``z_0 ... z_{N-1}``."""
        return torch.linspace(
            self.v_min, self.v_max, self.n_atoms, device=device, dtype=dtype
        )


def project_categorical_distribution(
    next_probs: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    discount: float | torch.Tensor,
    support: AtomSupport,
) -> torch.Tensor:
    """Project ``r + gamma^n * Z(s', a*)`` back onto the fixed atom support.

    This is the Bellemare et al. (2017) categorical projection Phi, extended
    with a per-sample discount so that the n-step returns of Algorithm 1 can be
    bootstrapped with ``gamma^n``.

    Args:
        next_probs: ``(batch, n_atoms)`` next-state return distribution, already
            selected for the bootstrap action (double Q-learning selects it with
            the online network and evaluates it with ``theta^-``).
        rewards: ``(batch,)`` n-step returns ``R_t``.
        dones: ``(batch,)`` terminal flags; terminal transitions are not
            bootstrapped.
        discount: scalar or ``(batch,)`` discount applied to the next-state
            atoms (``gamma ** n`` for n-step returns).
        support: the shared atom support.
    """
    if next_probs.ndim != 2:
        raise ValueError("next_probs must have shape (batch, n_atoms)")
    batch, n_atoms = next_probs.shape
    if n_atoms != support.n_atoms:
        raise ValueError(
            f"next_probs has {n_atoms} atoms, support has {support.n_atoms}"
        )
    rewards = rewards.reshape(-1)
    dones = dones.reshape(-1)
    if rewards.shape[0] != batch or dones.shape[0] != batch:
        raise ValueError("rewards and dones must have one entry per transition")

    device = next_probs.device
    dtype = next_probs.dtype
    atoms = support.atoms(device=device, dtype=dtype)

    if isinstance(discount, torch.Tensor):
        discounts = discount.reshape(-1).to(device=device, dtype=dtype)
        if discounts.shape[0] != batch:
            raise ValueError("discount tensor must have one entry per transition")
    else:
        discounts = torch.full((batch,), float(discount), device=device, dtype=dtype)

    rewards = rewards.to(device=device, dtype=dtype)
    continues = (1.0 - dones.to(device=device, dtype=dtype)).unsqueeze(1)
    target_z = rewards.unsqueeze(1) + continues * discounts.unsqueeze(
        1
    ) * atoms.unsqueeze(0)
    target_z = target_z.clamp(support.v_min, support.v_max)

    b = (target_z - support.v_min) / support.delta_z
    lower = b.floor().long()
    upper = b.ceil().long()
    # Keep the mass of atoms that land exactly on a support point: nudge the
    # degenerate l == u case apart so one of the two interpolation weights is 1.
    lower[(upper > 0) & (lower == upper)] -= 1
    upper[(lower < support.n_atoms - 1) & (lower == upper)] += 1

    projected = torch.zeros_like(next_probs)
    projected.scatter_add_(1, lower, next_probs * (upper.to(dtype) - b))
    projected.scatter_add_(1, upper, next_probs * (b - lower.to(dtype)))
    return projected


def categorical_kl_divergence(
    target_probs: torch.Tensor,
    predicted_log_probs: torch.Tensor,
    reduction: Reduction = "mean",
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """``D_KL(Z || Z(s, a; theta))`` between two categorical return distributions.

    The full divergence is returned (not just the cross-entropy): the entropy
    of the target is constant in ``theta``, so the gradients match the usual C51
    cross-entropy loss while the value stays zero for a perfect prediction,
    which makes the quantity readable as a training diagnostic.
    """
    if target_probs.shape != predicted_log_probs.shape:
        raise ValueError("target and prediction must have the same shape")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"unknown reduction {reduction!r}")

    zero = torch.zeros((), dtype=target_probs.dtype, device=target_probs.device)
    cross_entropy = torch.where(
        target_probs > 0, target_probs * predicted_log_probs, zero
    ).sum(dim=-1)
    target_entropy = torch.special.xlogy(target_probs, target_probs).sum(dim=-1)
    per_sample = target_entropy - cross_entropy

    if weights is not None:
        per_sample = per_sample * weights.reshape(-1).to(per_sample.dtype)

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    return per_sample


def combine_dueling_streams(
    value_logits: torch.Tensor, advantage_logits: torch.Tensor
) -> torch.Tensor:
    """``Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))``, per atom.

    Args:
        value_logits: ``(batch, 1, n_atoms)`` state-value stream.
        advantage_logits: ``(batch, n_actions, n_atoms)`` advantage stream.
    """
    if value_logits.ndim != 3 or value_logits.shape[1] != 1:
        raise ValueError("value_logits must have shape (batch, 1, n_atoms)")
    if advantage_logits.ndim != 3:
        raise ValueError("advantage_logits must have shape (batch, n_actions, n_atoms)")
    if value_logits.shape[0] != advantage_logits.shape[0]:
        raise ValueError("value and advantage streams must share the batch dimension")
    if value_logits.shape[2] != advantage_logits.shape[2]:
        raise ValueError("value and advantage streams must share the atom dimension")
    return value_logits + advantage_logits - advantage_logits.mean(dim=1, keepdim=True)
