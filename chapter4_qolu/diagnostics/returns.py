"""The realised returns a predicted return distribution is scored against.

This environment has **no terminal state inside a run**: ``done`` is raised only
when the iteration budget is exhausted, so there is no episode boundary at which
a return simply stops.  That leaves two honest readings of "the return that
actually happened", and this package reports both rather than choosing:

``n_step_reward_sum``
    ``sum_{k<n} gamma^k r_{t+k}``, truncated at the end of the run.  The
    bootstrap-free half of Algorithm 1's n-step return, at the agent's own
    ``n_step`` and ``gamma``.

``n_step_target``
    the same sum plus ``gamma^n * Q(s_{t+n}, argmax_a Q(s_{t+n}, a))`` evaluated
    with the agent's own target network - i.e. **exactly the quantity the
    network is regressed onto**.  It is the fairest statement of what the model
    was asked to predict, and the least independent: a model is being scored
    partly against its own bootstrap.

``tail_return``
    ``sum_{k} gamma^k r_{t+k}`` all the way to the end of the run, with no
    bootstrap anywhere.  This is the model-free ground truth and is the target
    the headline statistics use.

Design note (what "the next reward" means here).  One iteration produces one
appraisal per candidate in the decision buffer, and the candidates are redrawn
every iteration, so there is no per-parcel trajectory to follow.  The agent's
own replay buffer (:mod:`chapter4_qolu.agent.replay`) accumulates its n-step
returns over the order transitions are *pushed* in - candidate 0 to
``samples_per_iteration - 1`` of iteration ``t``, then the same for ``t + 1`` -
so that flattened appraisal stream is the sequence these functions discount
over too.  Reading it any other way would score the network against a return it
was never trained on.  The consequence is worth stating plainly: with
``gamma = 0.99`` and 50 appraisals per iteration, the effective horizon of
``tail_return`` is about 100 appraisals, i.e. roughly two iterations of the
simulation, not two iterations' worth of landscape change.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "TARGET_DESCRIPTIONS",
    "TARGET_NAMES",
    "n_step_reward_sums",
    "n_step_targets",
    "tail_returns",
]

#: The three realised-return readings, in reporting order.
TARGET_NAMES: tuple[str, ...] = (
    "n_step_reward_sum",
    "n_step_target",
    "tail_return",
)

TARGET_DESCRIPTIONS: dict[str, str] = {
    "n_step_reward_sum": (
        "bootstrap-free n-step reward sum at the agent's own n and gamma, "
        "truncated at the end of the run"
    ),
    "n_step_target": (
        "the agent's own training target: the n-step reward sum plus "
        "gamma^n times the double-Q bootstrap from its target network"
    ),
    "tail_return": (
        "bootstrap-free discounted tail of the appraisal stream, truncated at "
        "the end of the run"
    ),
}


def _as_stream(rewards: np.ndarray, name: str = "rewards") -> np.ndarray:
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional stream")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    return values


def _check_gamma(gamma: float) -> float:
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    return float(gamma)


def tail_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """``G_t = r_t + gamma * G_{t+1}``, with ``G`` past the last reward set to 0."""
    values = _as_stream(rewards)
    discount = _check_gamma(gamma)
    out = np.empty_like(values)
    running = 0.0
    for index in range(values.size - 1, -1, -1):
        running = values[index] + discount * running
        out[index] = running
    return out


def n_step_reward_sums(rewards: np.ndarray, gamma: float, n_step: int) -> np.ndarray:
    """``sum_{k<n} gamma^k r_{t+k}``, shortened where the run ends first."""
    values = _as_stream(rewards)
    discount = _check_gamma(gamma)
    steps = int(n_step)
    if steps < 1:
        raise ValueError("n_step must be positive")
    total = np.zeros_like(values)
    size = values.size
    for offset in range(min(steps, size)):
        total[: size - offset] += (discount**offset) * values[offset:]
    return total


def n_step_targets(
    rewards: np.ndarray,
    gamma: float,
    n_step: int,
    bootstrap_values: np.ndarray,
) -> np.ndarray:
    """The agent's own training target for every row of the appraisal stream.

    ``bootstrap_values[i]`` is ``Q(s', a*)`` for the transition stored at ``i``,
    i.e. the value of the state that transition *lands in*.  A window starting
    at ``i`` and running ``L`` steps therefore bootstraps from
    ``bootstrap_values[i + L - 1]`` discounted by ``gamma ** L``, which is what
    :class:`~chapter4_qolu.agent.replay.NStepReplayBuffer` stores.
    """
    values = _as_stream(rewards)
    bootstrap = _as_stream(bootstrap_values, name="bootstrap_values")
    if bootstrap.size != values.size:
        raise ValueError("bootstrap_values must have one entry per reward")
    discount = _check_gamma(gamma)
    steps = int(n_step)
    if steps < 1:
        raise ValueError("n_step must be positive")

    size = values.size
    indices = np.arange(size)
    lengths = np.minimum(steps, size - indices)
    last = indices + lengths - 1
    sums = n_step_reward_sums(values, discount, steps)
    return sums + (discount**lengths) * bootstrap[last]
