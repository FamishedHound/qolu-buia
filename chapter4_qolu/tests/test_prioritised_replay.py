"""Tests for prioritised experience replay (Schaul et al. 2016, proportional)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from chapter4_qolu.agent.prioritised_replay import (
    BetaSchedule,
    PrioritisedReplayBuffer,
    SumTree,
    importance_sampling_weights,
)
from chapter4_qolu.agent.replay import NStepReplayBuffer, Transition, n_step_return


def _obs(value: float) -> np.ndarray:
    return np.full((2, 3, 3), value, dtype=np.float32)


def _fill(buffer: NStepReplayBuffer, count: int, done_every: int = 0) -> None:
    for step in range(count):
        buffer.push(
            Transition(
                observation=_obs(step),
                action=step % 2,
                reward=float(step),
                next_observation=_obs(step + 1),
                done=bool(done_every) and step % done_every == done_every - 1,
            )
        )


class TestSumTree:
    def test_total_is_the_sum_of_the_leaves(self) -> None:
        tree = SumTree(8)
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0]):
            tree[index] = value
        assert tree.total == pytest.approx(10.0)

    def test_find_locates_the_leaf_holding_a_prefix_sum(self) -> None:
        tree = SumTree(4)
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0]):
            tree[index] = value
        # Cumulative boundaries: [0,1) -> 0, [1,3) -> 1, [3,6) -> 2, [6,10) -> 3.
        assert tree.find(0.5) == 0
        assert tree.find(1.5) == 1
        assert tree.find(3.5) == 2
        assert tree.find(9.5) == 3

    def test_updating_a_leaf_updates_the_total(self) -> None:
        tree = SumTree(4)
        tree[0] = 5.0
        tree[0] = 2.0
        assert tree.total == pytest.approx(2.0)

    def test_maximum_tracks_the_largest_leaf(self) -> None:
        tree = SumTree(4)
        tree[0] = 1.0
        tree[1] = 7.0
        tree[2] = 3.0
        assert tree.maximum == pytest.approx(7.0)

    def test_a_capacity_that_is_not_a_power_of_two_still_works(self) -> None:
        tree = SumTree(5)
        for index in range(5):
            tree[index] = 1.0
        assert tree.total == pytest.approx(5.0)
        assert tree.find(4.5) == 4

    def test_rejects_negative_priorities_and_bad_indices(self) -> None:
        tree = SumTree(4)
        with pytest.raises(ValueError):
            tree[0] = -1.0
        with pytest.raises(IndexError):
            tree[9] = 1.0

    def test_find_clamps_a_prefix_beyond_the_total(self) -> None:
        tree = SumTree(4)
        tree[0] = 1.0
        tree[1] = 1.0
        assert tree.find(10.0) in (0, 1)


class TestBetaSchedule:
    def test_starts_at_beta_start_and_ends_at_beta_end(self) -> None:
        schedule = BetaSchedule(start=0.4, end=1.0, steps=100)
        assert schedule(0) == pytest.approx(0.4)
        assert schedule(100) == pytest.approx(1.0)
        assert schedule(10**9) == pytest.approx(1.0)

    def test_anneals_linearly_in_between(self) -> None:
        schedule = BetaSchedule(start=0.4, end=1.0, steps=100)
        assert schedule(50) == pytest.approx(0.7)
        assert schedule(25) == pytest.approx(0.55)

    def test_is_monotone(self) -> None:
        schedule = BetaSchedule(start=0.4, end=1.0, steps=37)
        values = [schedule(step) for step in range(40)]
        assert all(b >= a for a, b in pairwise(values))

    def test_rejects_an_invalid_horizon(self) -> None:
        with pytest.raises(ValueError):
            BetaSchedule(start=0.4, end=1.0, steps=0)
        with pytest.raises(ValueError):
            BetaSchedule(start=1.2, end=1.0, steps=10)


class TestImportanceSamplingWeights:
    def test_the_largest_weight_is_normalised_to_one(self) -> None:
        probabilities = np.array([0.5, 0.25, 0.25])
        weights = importance_sampling_weights(probabilities, buffer_size=3, beta=0.6)
        assert float(weights.max()) == pytest.approx(1.0)

    def test_rarer_samples_earn_larger_weights(self) -> None:
        probabilities = np.array([0.5, 0.25, 0.125])
        weights = importance_sampling_weights(probabilities, buffer_size=3, beta=1.0)
        assert weights[2] > weights[1] > weights[0]

    def test_matches_the_published_formula(self) -> None:
        probabilities = np.array([0.4, 0.1])
        beta = 0.7
        raw = (2 * probabilities) ** (-beta)
        expected = raw / raw.max()
        np.testing.assert_allclose(
            importance_sampling_weights(probabilities, buffer_size=2, beta=beta),
            expected,
            rtol=1e-12,
        )

    def test_beta_zero_makes_every_weight_one(self) -> None:
        probabilities = np.array([0.7, 0.2, 0.1])
        weights = importance_sampling_weights(probabilities, buffer_size=3, beta=0.0)
        np.testing.assert_allclose(weights, np.ones(3))

    def test_rejects_non_positive_probabilities(self) -> None:
        with pytest.raises(ValueError):
            importance_sampling_weights(np.array([0.5, 0.0]), buffer_size=2, beta=0.5)


class TestNStepSemanticsArePreserved:
    """PER wraps the n-step buffer: the n-step transition is what is prioritised."""

    def test_emits_the_same_n_step_transitions_as_the_uniform_buffer(self) -> None:
        uniform = NStepReplayBuffer(capacity=32, n_step=3, gamma=0.5)
        prioritised = PrioritisedReplayBuffer(capacity=32, n_step=3, gamma=0.5)
        _fill(uniform, 12, done_every=5)
        _fill(prioritised, 12, done_every=5)
        assert len(uniform) == len(prioritised)
        # Uniform sampling is *with replacement*, so draw generously; the
        # prioritised draw of a full buffer at equal priorities is stratified
        # and therefore hits every slot exactly once.
        left = uniform.sample(500, np.random.default_rng(0))
        right = prioritised.sample(len(prioritised), np.random.default_rng(0))
        assert np.unique(right.rewards).size == len(prioritised)
        np.testing.assert_allclose(
            np.sort(np.unique(left.rewards)), np.sort(np.unique(right.rewards))
        )
        np.testing.assert_allclose(
            np.sort(np.unique(left.discounts)), np.sort(np.unique(right.discounts))
        )

    def test_aggregates_rewards_and_discounts_over_n_steps(self) -> None:
        buffer = PrioritisedReplayBuffer(capacity=10, n_step=3, gamma=0.5)
        rewards = [1.0, 2.0, 3.0]
        for step, reward in enumerate(rewards):
            buffer.push(Transition(_obs(step), step, reward, _obs(step + 1), False))
        batch = buffer.sample(1, np.random.default_rng(0))
        assert batch.rewards[0] == pytest.approx(n_step_return(rewards, gamma=0.5))
        assert batch.discounts[0] == pytest.approx(0.5**3)
        np.testing.assert_allclose(batch.observations[0], _obs(0))
        np.testing.assert_allclose(batch.next_observations[0], _obs(3))

    def test_capacity_evicts_the_oldest_transition(self) -> None:
        buffer = PrioritisedReplayBuffer(capacity=2, n_step=1, gamma=0.9)
        _fill(buffer, 5)
        assert len(buffer) == 2
        batch = buffer.sample(64, np.random.default_rng(0))
        assert set(batch.actions.tolist()) <= {0, 1}
        assert set(np.unique(batch.rewards).tolist()) == {3.0, 4.0}

    def test_can_sample_and_clear_behave_like_the_uniform_buffer(self) -> None:
        buffer = PrioritisedReplayBuffer(capacity=16, n_step=1, gamma=0.9)
        assert not buffer.can_sample(4)
        _fill(buffer, 8)
        assert buffer.can_sample(4)
        buffer.clear()
        assert len(buffer) == 0
        assert buffer.tree.total == pytest.approx(0.0)


class TestMaxPriorityInsertion:
    def test_a_new_transition_starts_at_the_largest_priority_seen(self) -> None:
        buffer = PrioritisedReplayBuffer(capacity=16, n_step=1, gamma=0.9, alpha=0.5)
        _fill(buffer, 4)
        # Every fresh transition starts at the same (initial) priority.
        assert buffer.priorities()[:4] == pytest.approx([1.0] * 4)
        buffer.update_priorities([0, 1, 2, 3], [0.1, 0.2, 9.0, 0.3])
        _fill(buffer, 1)
        assert buffer.priorities()[4] == pytest.approx(buffer.max_priority)
        assert buffer.max_priority == pytest.approx(9.0 + buffer.epsilon)

    def test_the_tree_stores_the_priority_raised_to_alpha(self) -> None:
        buffer = PrioritisedReplayBuffer(capacity=16, n_step=1, gamma=0.9, alpha=0.5)
        _fill(buffer, 2)
        buffer.update_priorities([0], [4.0])
        assert buffer.tree[0] == pytest.approx((4.0 + buffer.epsilon) ** 0.5)

    def test_priorities_are_the_kl_loss_plus_a_small_epsilon(self) -> None:
        buffer = PrioritisedReplayBuffer(
            capacity=16, n_step=1, gamma=0.9, alpha=1.0, epsilon=1e-3
        )
        _fill(buffer, 2)
        buffer.update_priorities([0, 1], [0.0, 2.0])
        # A zero KL still leaves a chance of being resampled.
        assert buffer.tree[0] == pytest.approx(1e-3)
        assert buffer.tree[1] == pytest.approx(2.0 + 1e-3)

    def test_rejects_negative_priorities_and_mismatched_lengths(self) -> None:
        buffer = PrioritisedReplayBuffer(capacity=16, n_step=1, gamma=0.9)
        _fill(buffer, 2)
        with pytest.raises(ValueError):
            buffer.update_priorities([0], [-1.0])
        with pytest.raises(ValueError):
            buffer.update_priorities([0, 1], [1.0])


class TestProportionalSampling:
    def test_sampling_is_proportional_to_priority(self) -> None:
        buffer = PrioritisedReplayBuffer(capacity=8, n_step=1, gamma=0.9, alpha=1.0)
        _fill(buffer, 4)
        buffer.update_priorities([0, 1, 2, 3], [7.0, 1.0, 1.0, 1.0])
        rng = np.random.default_rng(20260924)
        counts = np.zeros(4)
        for _ in range(200):
            batch = buffer.sample(8, rng)
            for index in batch.indices:
                counts[index] += 1
        share = counts / counts.sum()
        # Priorities 7:1:1:1 -> the first transition should take ~70% of draws.
        assert share[0] == pytest.approx(0.7, abs=0.06)
        assert all(share[index] == pytest.approx(0.1, abs=0.05) for index in (1, 2, 3))

    def test_alpha_zero_recovers_uniform_sampling(self) -> None:
        buffer = PrioritisedReplayBuffer(capacity=8, n_step=1, gamma=0.9, alpha=0.0)
        _fill(buffer, 4)
        buffer.update_priorities([0, 1, 2, 3], [7.0, 1.0, 1.0, 1.0])
        rng = np.random.default_rng(11)
        counts = np.zeros(4)
        for _ in range(200):
            for index in buffer.sample(8, rng).indices:
                counts[index] += 1
        share = counts / counts.sum()
        assert all(value == pytest.approx(0.25, abs=0.04) for value in share)

    def test_the_batch_carries_indices_probabilities_and_weights(self) -> None:
        buffer = PrioritisedReplayBuffer(capacity=16, n_step=1, gamma=0.9)
        _fill(buffer, 8)
        batch = buffer.sample(4, np.random.default_rng(0), beta=0.6)
        assert batch.indices.shape == (4,)
        assert batch.weights.shape == (4,)
        assert batch.probabilities.shape == (4,)
        assert float(batch.weights.max()) == pytest.approx(1.0)
        assert np.all(batch.weights > 0.0)
        assert batch.observations.shape == (4, 2, 3, 3)

    def test_importance_weights_offset_the_sampling_bias(self) -> None:
        buffer = PrioritisedReplayBuffer(capacity=8, n_step=1, gamma=0.9, alpha=1.0)
        _fill(buffer, 4)
        buffer.update_priorities([0, 1, 2, 3], [7.0, 1.0, 1.0, 1.0])
        batch = buffer.sample(64, np.random.default_rng(2), beta=1.0)
        by_index = dict(zip(batch.indices.tolist(), batch.weights.tolist()))
        assert by_index[0] < by_index[1]

    def test_beta_defaults_to_the_schedule(self) -> None:
        buffer = PrioritisedReplayBuffer(
            capacity=8, n_step=1, gamma=0.9, beta=BetaSchedule(0.4, 1.0, steps=4)
        )
        _fill(buffer, 4)
        assert buffer.beta_now() == pytest.approx(0.4)
        for _ in range(4):
            buffer.sample(2, np.random.default_rng(0))
        assert buffer.beta_now() == pytest.approx(1.0)

    def test_sampling_an_empty_buffer_raises(self) -> None:
        buffer = PrioritisedReplayBuffer(capacity=8, n_step=1, gamma=0.9)
        with pytest.raises(ValueError):
            buffer.sample(1, np.random.default_rng(0))

    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError):
            PrioritisedReplayBuffer(capacity=8, n_step=1, gamma=0.9, alpha=-0.1)
        with pytest.raises(ValueError):
            PrioritisedReplayBuffer(capacity=8, n_step=1, gamma=0.9, epsilon=-1.0)
