"""Consensus across top-down stakeholders.

"Once trained, each QOLU agent assigns a quality score to candidate
conversions.  Summing these scores across all stakeholders provides a
consensus-driven measure that balances ecological, economic, and policy-related
objectives."

The sum is unweighted as published, and that is what runs by default.  The
optional weighted reading - in which the scenario weights are the instrument
that does the *balancing* the sentence above describes - is available through
the ``weights`` argument of :func:`action_weighted_votes` and is selected by
:attr:`chapter4_qolu.config.QOLURewardConfig.weighted_consensus`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "action_weighted_votes",
    "balanced_weights",
    "consensus_decisions",
    "consensus_scores",
]

#: Index of the "approve" action in the QOLU action set (see env.urban_env).
_APPROVE = 1


def balanced_weights(weights: Sequence[float]) -> np.ndarray:
    """Stakeholder weights rescaled to mean 1.

    Design note: the consensus is a *sum*, so weighting it with raw scenario
    weights (which sum to about 1) would shrink the whole consensus score by a
    factor of ``n`` and move the decision threshold relative to the published
    rule.  Dividing by the mean keeps the scale of the sum and makes the
    published equal weighting an exact no-op - Scenario 1 makes the same
    decisions as the unweighted sum, bit for bit - while Scenario 3's 0.5
    woodland weight becomes the 1.5x vote that Table 3 implies.
    """
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("at least one stakeholder weight is required")
    if np.any(values < 0.0):
        raise ValueError("stakeholder weights must be non-negative")
    mean = float(values.mean())
    if mean <= 0.0:
        raise ValueError("stakeholder weights must not sum to zero")
    return values / mean


def consensus_scores(quality_scores: Sequence[np.ndarray]) -> np.ndarray:
    """Sum per-stakeholder quality scores for each candidate conversion."""
    scores = [np.asarray(score, dtype=np.float64) for score in quality_scores]
    if not scores:
        raise ValueError("at least one stakeholder is required")
    reference = scores[0].shape
    for score in scores[1:]:
        if score.shape != reference:
            raise ValueError(
                f"stakeholder score shapes differ: {reference} vs {score.shape}"
            )
    return np.sum(scores, axis=0)


def consensus_decisions(
    quality_scores: Sequence[np.ndarray], threshold: float = 0.0
) -> np.ndarray:
    """Approve a conversion when the summed quality score exceeds ``threshold``."""
    return (consensus_scores(quality_scores) > threshold).astype(np.int64)


def action_weighted_votes(
    quality_scores: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> list[np.ndarray]:
    """Turn epsilon-greedy actions into signed quality votes.

    The published gating rule is the plain sum of stakeholder quality scores,
    which is what :func:`consensus_decisions` implements and what should be used
    once the agents are trained.

    Design note (training only): during training an exploratory action has to be
    able to change what the environment does, otherwise the transition stored in
    the replay buffer would not match the action taken.  Each stakeholder
    therefore votes with the *magnitude* of its quality score and the *sign* of
    the action it actually chose.  With greedy actions the sign of the vote is
    the sign of the score, so this reduces exactly to the published rule at
    epsilon = 0.

    Design note (``weights``, off by default): the scenario weights read as the
    instrument that balances the three objectives, i.e. stakeholder ``i``
    contributes ``w_i`` times its vote.  ``weights`` are per stakeholder, in the
    same order as ``quality_scores``, and are normalised to mean 1 by
    :func:`balanced_weights` so that an equal weighting is an exact no-op.
    """
    scores = [np.asarray(score, dtype=np.float64) for score in quality_scores]
    taken = [np.asarray(action, dtype=np.int64) for action in actions]
    if len(scores) != len(taken):
        raise ValueError(
            f"got {len(scores)} score arrays for {len(taken)} action arrays"
        )
    if not scores:
        raise ValueError("at least one stakeholder is required")
    if weights is None:
        shares = np.ones(len(scores), dtype=np.float64)
    else:
        shares = balanced_weights(weights)
        if shares.size != len(scores):
            raise ValueError(
                f"got {shares.size} stakeholder weights for {len(scores)} "
                "score arrays"
            )
    votes: list[np.ndarray] = []
    for score, action, share in zip(scores, taken, shares):
        if score.shape != action.shape:
            raise ValueError(
                f"scores {score.shape} and actions {action.shape} must match"
            )
        sign = np.where(action == _APPROVE, 1.0, -1.0)
        vote = np.abs(score) * sign
        votes.append(vote if weights is None else vote * share)
    return votes
