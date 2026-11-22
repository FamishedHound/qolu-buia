"""Tests for the n-step replay buffer used by Algorithm 1."""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.agent.replay import NStepReplayBuffer, Transition, n_step_return


def _obs(value: float) -> np.ndarray:
    return np.full((2, 3, 3), value, dtype=np.float32)


class TestNStepReturn:
    def test_matches_the_discounted_sum_formula(self) -> None:
        assert n_step_return([1.0, 2.0, 3.0], gamma=0.5) == pytest.approx(
            1.0 + 1.0 + 0.75
        )

    def test_single_reward_is_undiscounted(self) -> None:
        assert n_step_return([4.0], gamma=0.99) == pytest.approx(4.0)

    def test_empty_sequence_is_zero(self) -> None:
        assert n_step_return([], gamma=0.99) == pytest.approx(0.0)


class TestBufferMechanics:
    def test_len_tracks_emitted_n_step_transitions(self) -> None:
        buffer = NStepReplayBuffer(capacity=100, n_step=3, gamma=0.9)
        for step in range(5):
            buffer.push(Transition(_obs(step), 0, 1.0, _obs(step + 1), False))
        # The first two pushes only fill the n-step window.
        assert len(buffer) == 3

    def test_capacity_evicts_oldest_transitions(self) -> None:
        buffer = NStepReplayBuffer(capacity=2, n_step=1, gamma=0.9)
        for step in range(5):
            buffer.push(
                Transition(_obs(step), step, float(step), _obs(step + 1), False)
            )
        assert len(buffer) == 2
        # Sampling is uniform *with replacement*, so draw generously to be sure
        # both surviving transitions are observed.
        batch = buffer.sample(64, rng=np.random.default_rng(0))
        assert set(batch.actions.tolist()) == {3, 4}

    def test_aggregates_rewards_and_bootstrap_state_over_n_steps(self) -> None:
        buffer = NStepReplayBuffer(capacity=10, n_step=3, gamma=0.5)
        rewards = [1.0, 2.0, 3.0]
        for step, reward in enumerate(rewards):
            buffer.push(Transition(_obs(step), step, reward, _obs(step + 1), False))
        batch = buffer.sample(1, rng=np.random.default_rng(0))
        assert batch.rewards[0] == pytest.approx(n_step_return(rewards, gamma=0.5))
        assert batch.discounts[0] == pytest.approx(0.5**3)
        np.testing.assert_allclose(batch.observations[0], _obs(0))
        np.testing.assert_allclose(batch.next_observations[0], _obs(3))
        assert batch.dones[0] == pytest.approx(0.0)

    def test_terminal_truncates_the_n_step_window(self) -> None:
        buffer = NStepReplayBuffer(capacity=10, n_step=3, gamma=0.5)
        buffer.push(Transition(_obs(0), 0, 1.0, _obs(1), False))
        buffer.push(Transition(_obs(1), 1, 2.0, _obs(2), True))
        assert len(buffer) == 2
        batch = buffer.sample(64, rng=np.random.default_rng(0))
        by_action = {
            int(action): (float(reward), float(discount), float(done))
            for action, reward, discount, done in zip(
                batch.actions, batch.rewards, batch.discounts, batch.dones
            )
        }
        assert set(by_action) == {0, 1}
        # The window that started at t=0 is flushed as a truncated 2-step return.
        assert by_action[0] == (
            pytest.approx(1.0 + 0.5 * 2.0),
            pytest.approx(0.5**2),
            1.0,
        )
        assert by_action[1] == (pytest.approx(2.0), pytest.approx(0.5), 1.0)

    def test_episode_boundary_resets_the_window(self) -> None:
        buffer = NStepReplayBuffer(capacity=10, n_step=3, gamma=0.9)
        buffer.push(Transition(_obs(0), 0, 1.0, _obs(1), True))
        buffer.push(Transition(_obs(1), 1, 5.0, _obs(2), False))
        batch = buffer.sample(len(buffer), rng=np.random.default_rng(0))
        rewards = {int(a): float(r) for a, r in zip(batch.actions, batch.rewards)}
        assert rewards[0] == pytest.approx(1.0)

    def test_sample_returns_correctly_shaped_arrays(self) -> None:
        buffer = NStepReplayBuffer(capacity=50, n_step=1, gamma=0.99)
        for step in range(20):
            buffer.push(Transition(_obs(step), step % 2, 1.0, _obs(step + 1), False))
        batch = buffer.sample(8, rng=np.random.default_rng(1))
        assert batch.observations.shape == (8, 2, 3, 3)
        assert batch.next_observations.shape == (8, 2, 3, 3)
        assert batch.actions.shape == (8,)
        assert batch.rewards.shape == (8,)
        assert batch.dones.shape == (8,)
        assert batch.discounts.shape == (8,)
        assert batch.observations.dtype == np.float32

    def test_can_sample_reports_readiness(self) -> None:
        buffer = NStepReplayBuffer(capacity=50, n_step=1, gamma=0.99)
        assert not buffer.can_sample(4)
        for step in range(4):
            buffer.push(Transition(_obs(step), 0, 0.0, _obs(step + 1), False))
        assert buffer.can_sample(4)

    def test_sampling_an_empty_buffer_raises(self) -> None:
        buffer = NStepReplayBuffer(capacity=10, n_step=1, gamma=0.99)
        with pytest.raises(ValueError):
            buffer.sample(1, rng=np.random.default_rng(0))

    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError):
            NStepReplayBuffer(capacity=0, n_step=1, gamma=0.99)
        with pytest.raises(ValueError):
            NStepReplayBuffer(capacity=10, n_step=0, gamma=0.99)
