"""Tests for the epsilon-greedy exploration schedules.

Table 1 of the paper fixes epsilon-greedy exploration but explicitly leaves the
schedule unspecified, so both a linear and an exponential decay are provided.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from chapter4_qolu.agent.exploration import (
    ExponentialEpsilonSchedule,
    LinearEpsilonSchedule,
)


class TestLinearEpsilonSchedule:
    def test_starts_at_the_initial_value(self) -> None:
        schedule = LinearEpsilonSchedule(start=1.0, end=0.05, decay_steps=1000)
        assert schedule(0) == pytest.approx(1.0)

    def test_reaches_the_final_value_at_decay_steps(self) -> None:
        schedule = LinearEpsilonSchedule(start=1.0, end=0.05, decay_steps=1000)
        assert schedule(1000) == pytest.approx(0.05)

    def test_is_linear_at_the_midpoint(self) -> None:
        schedule = LinearEpsilonSchedule(start=1.0, end=0.0, decay_steps=100)
        assert schedule(50) == pytest.approx(0.5)

    def test_clamps_after_the_decay_window(self) -> None:
        schedule = LinearEpsilonSchedule(start=1.0, end=0.05, decay_steps=10)
        assert schedule(10_000) == pytest.approx(0.05)

    def test_is_monotonically_non_increasing(self) -> None:
        schedule = LinearEpsilonSchedule(start=0.9, end=0.1, decay_steps=37)
        values = [schedule(step) for step in range(60)]
        assert all(later <= earlier + 1e-12 for earlier, later in pairwise(values))

    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError):
            LinearEpsilonSchedule(start=0.1, end=0.5, decay_steps=10)
        with pytest.raises(ValueError):
            LinearEpsilonSchedule(start=1.0, end=0.0, decay_steps=0)
        with pytest.raises(ValueError):
            LinearEpsilonSchedule(start=1.5, end=0.0, decay_steps=10)


class TestExponentialEpsilonSchedule:
    def test_starts_at_the_initial_value(self) -> None:
        schedule = ExponentialEpsilonSchedule(start=1.0, end=0.05, decay_steps=500)
        assert schedule(0) == pytest.approx(1.0)

    def test_decays_towards_the_final_value(self) -> None:
        schedule = ExponentialEpsilonSchedule(start=1.0, end=0.05, decay_steps=500)
        # epsilon(t) = end + (start - end) * exp(-t / decay_steps)
        assert schedule(500) == pytest.approx(0.05 + 0.95 * math.exp(-1.0))
        assert schedule(10_000) == pytest.approx(0.05, abs=1e-6)

    def test_is_monotonically_non_increasing(self) -> None:
        schedule = ExponentialEpsilonSchedule(start=1.0, end=0.01, decay_steps=25)
        values = [schedule(step) for step in range(100)]
        assert all(later <= earlier + 1e-12 for earlier, later in pairwise(values))

    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError):
            ExponentialEpsilonSchedule(start=0.1, end=0.5, decay_steps=10)
        with pytest.raises(ValueError):
            ExponentialEpsilonSchedule(start=1.0, end=0.0, decay_steps=-1)
