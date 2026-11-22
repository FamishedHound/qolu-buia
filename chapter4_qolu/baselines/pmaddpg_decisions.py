"""From P-MADDPG approve-scores to a decision: exploration, then consensus.

Split out of :mod:`chapter4_qolu.baselines.pmaddpg` so that the method module
holds Chapter 3's Algorithm 1 and nothing else.  Two steps live here, in the
order they are applied:

1. **exploration** - Chapter 3's published epsilon-greedy (line 392, "Our
   exploration and exploitation $\\epsilon$ coefficient is equal to $0.3$"), or
   DDPG's Ornstein-Uhlenbeck noise when ``epsilon_greedy`` is ``None``;
2. **consensus** - the paper's own gating rule, reached through
   :mod:`chapter4_qolu.agent.consensus` so that a baseline run and a
   ``train_joint`` run take literally the same code path.

Everything is a free function over arrays: nothing here holds state, so the
method keeps owning its generators and its noise processes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from chapter4_qolu.agent.consensus import action_weighted_votes, consensus_decisions
from chapter4_qolu.baselines.config import PMADDPGConfig
from chapter4_qolu.baselines.noise import OrnsteinUhlenbeckNoise

__all__ = [
    "appraisal_consensus",
    "epsilon_greedy_scores",
    "ornstein_uhlenbeck_scores",
    "resized_noise",
]


def epsilon_greedy_scores(
    values: np.ndarray, epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    """Chapter 3's published exploration, applied per candidate parcel.

    Design note: Chapter 3 gives the coefficient and nothing else, so the two
    free readings are taken the way a *property-based* problem suggests.  The
    unit an action applies to is a cell, so the coin is tossed per candidate
    parcel rather than once per iteration; and the "random action" of a
    continuous approve-score is a uniform draw from the score's own range,
    [0, 1].  One :meth:`~numpy.random.Generator.random` call supplies the coin
    and one the replacement, so the stream advances by exactly ``2 * size``
    whatever epsilon is - which keeps a run reproducible from its seed.
    """
    if epsilon <= 0.0:
        return values
    explore = rng.random(values.size) < float(epsilon)
    return np.where(explore, rng.random(values.size), values)


def resized_noise(
    noise: OrnsteinUhlenbeckNoise, size: int, config: PMADDPGConfig
) -> OrnsteinUhlenbeckNoise:
    """``noise`` if it is already ``size`` wide, otherwise a fresh process.

    The process has one component per parcel: a single size-1 process drawn
    once per parcel would make the perturbation one mean-reverting offset
    shared by the whole decision buffer rather than per-parcel exploration
    noise.  The generator is carried over, so resizing does not restart the
    stream.
    """
    if noise.size == int(size):
        return noise
    return OrnsteinUhlenbeckNoise(
        size=int(size),
        theta=config.ou_theta,
        sigma=config.ou_sigma,
        dt=config.ou_dt,
        rng=noise.rng,
    )


def ornstein_uhlenbeck_scores(
    values: np.ndarray, noise: OrnsteinUhlenbeckNoise, scale: float
) -> np.ndarray:
    """The DDPG alternative, used when ``epsilon_greedy`` is ``None``."""
    return values + float(scale) * noise.sample()


def appraisal_consensus(
    scores: Sequence[np.ndarray],
    threshold: float = 0.5,
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Turn per-stakeholder approve-scores into the published consensus decision.

    The paper's gating rule is the sum of stakeholder *quality scores*; here a
    stakeholder's quality score is its approve-score centred on the decision
    threshold, so "score above the threshold" means "in favour".  Routing the
    result through :func:`~chapter4_qolu.agent.consensus.action_weighted_votes`
    keeps this method on exactly the same code path as ``train_joint``, down to
    the optional scenario weighting of ``weights`` (``None`` = the published
    plain sum).
    """
    quality = [
        np.asarray(score, dtype=np.float64).reshape(-1) - threshold for score in scores
    ]
    actions = [(value >= 0.0).astype(np.int64) for value in quality]
    return consensus_decisions(action_weighted_votes(quality, actions, weights=weights))
