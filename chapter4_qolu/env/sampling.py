"""Decision-buffer sampling (Eq. 2 of the paper).

"A decision buffer is constructed to hold the parcels from which they are
sampled.  The probability of a parcel being sampled is directly proportional to
its attractiveness rating: P(sampled | A_i) ~ A_i", and "during each iteration,
50 parcels are sampled with replacement".
"""

from __future__ import annotations

import numpy as np

from chapter4_qolu.env.parcel_grid import ParcelIndex

__all__ = ["attractiveness_probabilities", "sample_decision_buffer"]


def attractiveness_probabilities(
    scores: np.ndarray, eligible: np.ndarray | None = None
) -> np.ndarray:
    """Normalise attractiveness scores into a sampling distribution.

    Design note: when every eligible parcel has zero attractiveness the
    distribution falls back to uniform over the eligible parcels, which keeps
    the simulation running on landscapes with no desirable features at all
    (the paper does not discuss this degenerate case).
    """
    weights = np.asarray(scores, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("scores must be a 2-D array")
    if not np.all(np.isfinite(weights)):
        raise ValueError("scores contain non-finite values")
    if float(weights.min(initial=0.0)) < 0.0:
        raise ValueError("attractiveness scores must be non-negative")

    if eligible is None:
        mask = np.ones(weights.shape, dtype=bool)
    else:
        mask = np.asarray(eligible, dtype=bool)
        if mask.shape != weights.shape:
            raise ValueError(
                f"eligibility mask shape {mask.shape} does not match scores {weights.shape}"
            )
    if not mask.any():
        raise ValueError("no eligible parcels to sample from")

    weights = np.where(mask, weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        uniform = mask.astype(np.float64)
        return uniform / float(uniform.sum())
    return weights / total


def sample_decision_buffer(
    scores: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    eligible: np.ndarray | None = None,
) -> tuple[ParcelIndex, ...]:
    """Draw ``n_samples`` parcels **with replacement**, P(parcel) ~ attractiveness."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    probabilities = attractiveness_probabilities(scores, eligible=eligible)
    width = probabilities.shape[1]
    flat = probabilities.reshape(-1)
    drawn = rng.choice(flat.size, size=n_samples, replace=True, p=flat)
    return tuple(
        ParcelIndex(int(index // width), int(index % width)) for index in drawn
    )
