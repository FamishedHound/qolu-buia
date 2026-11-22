"""Tests for the QOLU agent (Algorithm 1 + Table 1)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chapter4_qolu.agent.consensus import consensus_decisions
from chapter4_qolu.agent.qolu_agent import QOLUAgent, QOLUConfig


def make_agent(**overrides: object) -> QOLUAgent:
    settings: dict[str, object] = {
        "observation_channels": 4,
        "observation_window": 5,
        "n_actions": 2,
        "batch_size": 8,
        "learning_starts": 8,
        "replay_capacity": 200,
        "target_sync_interval": 4,
        "n_step": 2,
    }
    settings.update(overrides)
    return QOLUAgent(QOLUConfig(**settings), seed=0)  # type: ignore[arg-type]


def observations(count: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((count, 4, 5, 5)).astype(np.float32)


class TestPublishedHyperparameters:
    def test_defaults_match_table_1(self) -> None:
        config = QOLUConfig(observation_channels=11, observation_window=9)
        assert config.n_atoms == 51
        assert config.v_min == pytest.approx(-200.0)
        assert config.v_max == pytest.approx(200.0)
        assert config.batch_size == 256
        assert config.gamma == pytest.approx(0.99)
        assert config.conv_channels == (32, 64, 128)
        assert config.hidden_units == 256

    def test_uses_adam(self) -> None:
        agent = make_agent()
        assert isinstance(agent.optimizer, torch.optim.Adam)

    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError):
            QOLUConfig(observation_channels=0, observation_window=9)
        with pytest.raises(ValueError):
            QOLUConfig(observation_channels=4, observation_window=9, gamma=1.5)


class TestActionSelection:
    def test_returns_one_action_per_observation(self) -> None:
        agent = make_agent()
        actions = agent.act(observations(6), epsilon=0.0)
        assert actions.shape == (6,)
        assert set(np.unique(actions)).issubset({0, 1})

    def test_greedy_actions_match_the_argmax_of_q_values(self) -> None:
        agent = make_agent()
        obs = observations(5)
        greedy = agent.act(obs, epsilon=0.0)
        expected = agent.q_values(obs).argmax(axis=1)
        np.testing.assert_array_equal(greedy, expected)

    def test_epsilon_one_explores(self) -> None:
        agent = make_agent()
        obs = observations(200)
        greedy = agent.act(obs, epsilon=0.0)
        exploratory = agent.act(obs, epsilon=1.0)
        assert np.mean(greedy == exploratory) < 0.9

    def test_quality_scores_are_the_approve_minus_reject_advantage(self) -> None:
        agent = make_agent()
        obs = observations(4)
        q_values = agent.q_values(obs)
        np.testing.assert_allclose(
            agent.quality_scores(obs), q_values[:, 1] - q_values[:, 0], rtol=1e-5
        )

    def test_action_selection_does_not_track_gradients(self) -> None:
        agent = make_agent()
        agent.act(observations(3), epsilon=0.0)
        assert all(p.grad is None for p in agent.online_network.parameters())

    def test_epsilon_schedule_is_used_when_no_epsilon_is_given(self) -> None:
        agent = make_agent()
        assert agent.epsilon(0) == pytest.approx(agent.config.epsilon_start)
        assert agent.epsilon(10**9) == pytest.approx(agent.config.epsilon_end)


class TestLearning:
    def test_no_update_before_learning_starts(self) -> None:
        agent = make_agent()
        agent.remember(observations(1)[0], 1, 0.5, observations(1)[0], False)
        assert agent.learn() is None

    def test_learn_returns_a_finite_loss_and_changes_parameters(self) -> None:
        agent = make_agent()
        obs = observations(40)
        for index in range(39):
            agent.remember(
                obs[index], index % 2, float(index % 3), obs[index + 1], False
            )
        before = [p.detach().clone() for p in agent.online_network.parameters()]
        loss = agent.learn()
        assert loss is not None
        assert np.isfinite(loss)
        after = list(agent.online_network.parameters())
        assert any(not torch.allclose(a, b) for a, b in zip(before, after))

    def test_target_network_starts_synchronised_and_is_frozen(self) -> None:
        agent = make_agent()
        for online, target in zip(
            agent.online_network.parameters(), agent.target_network.parameters()
        ):
            assert torch.allclose(online, target)
            assert not target.requires_grad

    def test_target_network_is_resynchronised_every_t_n_steps(self) -> None:
        agent = make_agent(target_sync_interval=2)
        obs = observations(40)
        for index in range(39):
            agent.remember(obs[index], index % 2, 1.0, obs[index + 1], False)
        agent.learn()
        drifted = any(
            not torch.allclose(online, target)
            for online, target in zip(
                agent.online_network.parameters(), agent.target_network.parameters()
            )
        )
        assert drifted
        agent.learn()  # second gradient step triggers the sync
        for online, target in zip(
            agent.online_network.parameters(), agent.target_network.parameters()
        ):
            assert torch.allclose(online, target)

    def test_double_q_learning_selects_with_online_and_evaluates_with_target(
        self,
    ) -> None:
        agent = make_agent()
        next_obs = torch.from_numpy(observations(4))
        selected = agent.select_bootstrap_actions(next_obs)
        expected = agent.online_network.q_values(next_obs).argmax(dim=1)
        torch.testing.assert_close(selected, expected)

    def test_gradient_clipping_is_configurable(self) -> None:
        agent = make_agent(max_grad_norm=0.01)
        obs = observations(40)
        for index in range(39):
            agent.remember(obs[index], index % 2, 10.0, obs[index + 1], False)
        assert agent.learn() is not None

    def test_state_dict_round_trip(self) -> None:
        agent = make_agent()
        obs = observations(2)
        before = agent.q_values(obs)
        restored = make_agent()
        restored.load_state_dict(agent.state_dict())
        np.testing.assert_allclose(restored.q_values(obs), before, rtol=1e-5)


class TestConsensus:
    def test_sums_stakeholder_quality_scores(self) -> None:
        scores = [np.array([1.0, -2.0]), np.array([0.5, 0.5])]
        decisions = consensus_decisions(scores)
        np.testing.assert_array_equal(decisions, np.array([1, 0]))

    def test_threshold_is_configurable(self) -> None:
        scores = [np.array([0.4, 0.8])]
        np.testing.assert_array_equal(
            consensus_decisions(scores, threshold=0.5), np.array([0, 1])
        )

    def test_requires_matching_shapes(self) -> None:
        with pytest.raises(ValueError):
            consensus_decisions([np.zeros(3), np.zeros(4)])

    def test_requires_at_least_one_stakeholder(self) -> None:
        with pytest.raises(ValueError):
            consensus_decisions([])
