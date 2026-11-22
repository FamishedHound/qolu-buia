"""Sampling a set of parcels from BUIA's categorical proposal distribution.

Algorithm 2 samples *one* action from ``Categorical(softmax(Z))`` per decision
point, while Section 3.5 allows an iteration to convert up to ten parcels.

Design note: the two are reconciled by drawing the iteration's proposals
*sequentially without replacement* from the same categorical distribution -
draw a parcel, remove it from the support, renormalise, draw again.  This is
the Plackett-Luce model, it keeps Algorithm 2's softmax exactly as published
for the first draw, and its log-probability stays differentiable, which is what
the REINFORCE update needs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

__all__ = ["sample_without_replacement"]


def sample_without_replacement(
    logits: torch.Tensor,
    k: int,
    rng: np.random.Generator,
    forced_indices: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], torch.Tensor]:
    """Draw ``k`` distinct items and return them with their joint log-probability.

    Args:
        logits: ``(n,)`` proposal logits ``Z``; ``-inf`` masks an item out.
        k: how many items to draw (clipped to the number of available items).
        rng: numpy generator, used for the draws themselves so that runs stay
            reproducible from a single seed.
        forced_indices: evaluate the log-probability of this specific ordered
            selection instead of sampling (used by tests and by policy updates
            that re-score an earlier proposal).

    Returns:
        The selected indices in draw order and the differentiable log
        probability ``sum_t log p(i_t | i_1..i_{t-1})``.
    """
    if logits.ndim != 1:
        raise ValueError("logits must be a 1-D tensor of per-candidate scores")
    if k < 1:
        raise ValueError("k must be positive")

    n_items = int(logits.shape[0])
    if n_items == 0:
        raise ValueError("cannot sample from an empty candidate set")

    available = torch.isfinite(logits)
    n_available = int(available.sum().item())
    if n_available == 0:
        raise ValueError("every candidate is masked out")

    draws = min(k, n_available)
    if forced_indices is not None:
        selection = [int(index) for index in forced_indices]
        if len(set(selection)) != len(selection):
            raise ValueError("forced_indices must not repeat an item")
        if any(not 0 <= index < n_items for index in selection):
            raise ValueError("forced_indices out of range")
        draws = len(selection)
    else:
        selection = []

    log_probability = logits.new_zeros(())
    chosen: list[int] = []

    for position in range(draws):
        # A fresh additive mask each round: masking in place would bump the
        # version counter of a tensor autograd has already saved, which breaks
        # the backward pass of the REINFORCE update.
        bias = torch.zeros_like(logits, requires_grad=False)
        if chosen:
            bias[chosen] = -float("inf")
        log_probabilities = torch.log_softmax(logits + bias, dim=0)
        if forced_indices is None:
            probabilities = log_probabilities.detach().exp().cpu().numpy()
            probabilities = np.clip(probabilities, 0.0, None)
            total = float(probabilities.sum())
            if total <= 0.0:  # pragma: no cover - numerical safety net
                break
            index = int(rng.choice(probabilities.size, p=probabilities / total))
        else:
            index = selection[position]
        log_probability = log_probability + log_probabilities[index]
        chosen.append(index)

    return tuple(chosen), log_probability
