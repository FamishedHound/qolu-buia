"""Epsilon-greedy exploration schedules.

Table 1 of the paper specifies "epsilon-greedy (schedule not fixed)", so both a
linear and an exponential decay are provided; the linear one is the default
used by :class:`chapter4_qolu.config.QOLUConfig`.

Algorithm 1 line 5 selects actions "by exploiting noisy network parameters
theta, **or** exploratory action based on Epsilon", and Table 1 lists
epsilon-greedy as the exploration row.  Design note: the line keeps *both*
mechanisms rather than choosing between them, so the schedules here are applied
on top of the NoisyNet layers of :mod:`chapter4_qolu.agent.noisy` (which are
what the greedy branch exploits).  ``QOLUConfig.noisy_nets=False`` leaves
epsilon-greedy as the only exploration mechanism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["EpsilonSchedule", "ExponentialEpsilonSchedule", "LinearEpsilonSchedule"]


@dataclass(frozen=True)
class EpsilonSchedule:
    """Base class validating the shared schedule parameters."""

    start: float = 1.0
    end: float = 0.05
    decay_steps: int = 10_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.end <= self.start <= 1.0:
            raise ValueError("require 0 <= end <= start <= 1")
        if self.decay_steps < 1:
            raise ValueError("decay_steps must be positive")

    def __call__(self, step: int) -> float:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class LinearEpsilonSchedule(EpsilonSchedule):
    """Linear decay from ``start`` to ``end`` over ``decay_steps`` steps."""

    def __call__(self, step: int) -> float:
        if step <= 0:
            return float(self.start)
        if step >= self.decay_steps:
            return float(self.end)
        progress = step / self.decay_steps
        return float(self.start + progress * (self.end - self.start))


@dataclass(frozen=True)
class ExponentialEpsilonSchedule(EpsilonSchedule):
    """Exponential decay ``end + (start - end) * exp(-step / decay_steps)``."""

    def __call__(self, step: int) -> float:
        if step <= 0:
            return float(self.start)
        return float(
            self.end + (self.start - self.end) * math.exp(-step / self.decay_steps)
        )
