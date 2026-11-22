"""Tests for the two Rainbow components of QOLU (Hessel et al. 2018).

Algorithm 1 gives C51, n-step returns, double Q-learning and a dueling head.
This module covers the NoisyNet layers of Algorithm 1 line 5 - which the
chapter *does* state, so they are on by default - and prioritised replay, a
Rainbow component the chapter never mentions, which is therefore an opt-in.
It also covers the two switches that keep the plain C51 reading reachable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from chapter4_qolu.agent.networks import QOLUNetwork, QOLUNetworkConfig
from chapter4_qolu.agent.noisy import NoisyLinear
from chapter4_qolu.agent.prioritised_replay import PrioritisedReplayBuffer
from chapter4_qolu.agent.qolu_agent import QOLUAgent, QOLUConfig
from chapter4_qolu.agent.replay import NStepReplayBuffer

C51_ONLY: dict[str, object] = {"noisy_nets": False, "prioritised_replay": False}


def make_agent(**overrides: object) -> QOLUAgent:
    settings: dict[str, object] = {
        "observation_channels": 4,
        "observation_window": 5,
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


def fill(agent: QOLUAgent, count: int = 40) -> np.ndarray:
    obs = np.random.default_rng(1).random((count + 1, 4, 5, 5)).astype(np.float32)
    for index in range(count):
        agent.remember(obs[index], index % 2, float(index % 3), obs[index + 1], False)
    return obs


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
class TestRainbowDefaults:
    def test_the_defaults_are_the_chapters_reading(self) -> None:
        # Algorithm 1 line 5 states the noisy parameters, so they are on; the
        # chapter never mentions prioritisation, so PER is off.
        config = QOLUConfig(observation_channels=11, observation_window=9)
        assert config.noisy_nets is True
        assert config.prioritised_replay is False

    def test_published_noisy_and_per_hyperparameters(self) -> None:
        config = QOLUConfig(observation_channels=11, observation_window=9)
        assert config.noisy_sigma0 == pytest.approx(0.5)
        assert config.per_alpha == pytest.approx(0.5)
        assert config.per_beta_start == pytest.approx(0.4)
        assert config.per_beta_end == pytest.approx(1.0)
        assert config.per_beta_steps > 0

    def test_epsilon_greedy_is_retained_on_top_of_the_noisy_net(self) -> None:
        # Design note under test: Algorithm 1 line 5 keeps *both* mechanisms.
        config = QOLUConfig(observation_channels=11, observation_window=9)
        assert config.epsilon_start == pytest.approx(1.0)
        assert config.epsilon_end == pytest.approx(0.05)

    def test_rejects_invalid_rainbow_settings(self) -> None:
        for bad in (
            {"noisy_sigma0": -0.1},
            {"per_alpha": -0.1},
            {"per_beta_start": 1.2},
            {"per_beta_end": -0.1},
            {"per_beta_steps": 0},
            {"per_epsilon": -1.0},
            {"noisy_scope": "nonsense"},
        ):
            with pytest.raises(ValueError):
                QOLUConfig(observation_channels=4, observation_window=5, **bad)


# ----------------------------------------------------------------------
# the network
# ----------------------------------------------------------------------
class TestNoisyNetwork:
    def test_the_streams_are_noisy_by_default(self) -> None:
        net = QOLUNetwork(QOLUNetworkConfig(in_channels=4, window=5, n_actions=2))
        assert isinstance(net.value_head, NoisyLinear)
        assert isinstance(net.advantage_head, NoisyLinear)

    def test_the_default_scope_makes_every_fully_connected_layer_noisy(self) -> None:
        # Hessel et al. "replace all linear layers with their noisy equivalent".
        net = QOLUNetwork(QOLUNetworkConfig(in_channels=4, window=5, n_actions=2))
        linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
        assert linears == []
        assert len([m for m in net.modules() if isinstance(m, NoisyLinear)]) == 5

    def test_the_streams_only_scope_leaves_the_trunk_deterministic(self) -> None:
        net = QOLUNetwork(
            QOLUNetworkConfig(
                in_channels=4, window=5, n_actions=2, noisy_scope="streams"
            )
        )
        assert isinstance(net.value_head, NoisyLinear)
        assert isinstance(net.advantage_head, NoisyLinear)
        trunk = [m for m in net.residual_mlp.modules() if isinstance(m, NoisyLinear)]
        assert trunk == []
        assert type(net.projection[0]) is nn.Linear

    def test_the_convolutions_are_never_noisy(self) -> None:
        net = QOLUNetwork(QOLUNetworkConfig(in_channels=4, window=5, n_actions=2))
        assert [type(m).__name__ for m in net.conv if isinstance(m, nn.Conv2d)] == [
            "Conv2d"
        ] * 3

    def test_reset_noise_changes_the_forward_pass(self) -> None:
        torch.manual_seed(0)
        net = QOLUNetwork(QOLUNetworkConfig(in_channels=4, window=5, n_actions=2))
        batch = torch.randn(3, 4, 5, 5)
        net.reset_noise()
        first = net(batch)
        net.reset_noise()
        assert not torch.allclose(first, net(batch))

    def test_zero_noise_is_reproducible(self) -> None:
        torch.manual_seed(1)
        net = QOLUNetwork(QOLUNetworkConfig(in_channels=4, window=5, n_actions=2))
        batch = torch.randn(3, 4, 5, 5)
        net.zero_noise()
        first = net(batch)
        net.reset_noise()
        net.zero_noise()
        torch.testing.assert_close(net(batch), first)

    def test_the_head_still_produces_51_atoms_per_action(self) -> None:
        net = QOLUNetwork(QOLUNetworkConfig(in_channels=4, window=5, n_actions=2))
        assert net(torch.zeros(3, 4, 5, 5)).shape == (3, 2, 51)
        probs = net.distribution(torch.zeros(3, 4, 5, 5))
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(3, 2))

    def test_gradients_reach_every_noisy_parameter(self) -> None:
        torch.manual_seed(2)
        net = QOLUNetwork(QOLUNetworkConfig(in_channels=4, window=5, n_actions=2))
        net.reset_noise()
        net(torch.randn(2, 4, 5, 5)).square().sum().backward()
        assert [name for name, p in net.named_parameters() if p.grad is None] == []


# ----------------------------------------------------------------------
# the agent with Rainbow on
# ----------------------------------------------------------------------
class TestNoisyAgent:
    def test_acting_resamples_the_noise_every_call(self) -> None:
        agent = make_agent()
        obs = observations(3)
        agent.act(obs, epsilon=0.0)
        first = agent.online_network.value_head.weight_epsilon.clone()
        agent.act(obs, epsilon=0.0)
        assert not torch.allclose(first, agent.online_network.value_head.weight_epsilon)

    def test_the_quality_score_is_read_off_the_mean_parameters(self) -> None:
        # The consensus metric has to be a *score*, not an exploratory draw.
        agent = make_agent()
        obs = observations(4)
        first = agent.quality_scores(obs)
        agent.act(obs, epsilon=0.0)
        np.testing.assert_allclose(agent.quality_scores(obs), first, rtol=1e-6)

    def test_epsilon_greedy_is_still_applied_on_top(self) -> None:
        agent = make_agent()
        obs = observations(200)
        greedy = agent.act(obs, epsilon=0.0)
        exploratory = agent.act(obs, epsilon=1.0)
        assert np.mean(greedy == exploratory) < 0.9

    def test_learning_resamples_the_noise_on_both_networks(self) -> None:
        agent = make_agent()
        fill(agent)
        agent.learn()
        online = agent.online_network.value_head.weight_epsilon.clone()
        target = agent.target_network.value_head.weight_epsilon.clone()
        agent.learn()
        assert not torch.allclose(
            online, agent.online_network.value_head.weight_epsilon
        )
        assert not torch.allclose(
            target, agent.target_network.value_head.weight_epsilon
        )

    def test_the_sigma_parameters_are_trained(self) -> None:
        agent = make_agent()
        fill(agent)
        before = agent.online_network.value_head.weight_sigma.detach().clone()
        for _ in range(3):
            agent.learn()
        after = agent.online_network.value_head.weight_sigma.detach()
        assert not torch.allclose(before, after)

    def test_state_dict_round_trip_carries_the_noisy_parameters(self) -> None:
        agent = make_agent()
        fill(agent)
        agent.learn()
        restored = make_agent()
        restored.load_state_dict(agent.state_dict())
        restored.online_network.zero_noise()
        agent.online_network.zero_noise()
        obs = observations(3)
        np.testing.assert_allclose(
            restored.q_values(obs), agent.q_values(obs), rtol=1e-6
        )


def make_per_agent(**overrides: object) -> QOLUAgent:
    """An agent with the opt-in prioritised buffer (the ``qolu_buia_per`` key)."""
    settings: dict[str, object] = {"prioritised_replay": True}
    settings.update(overrides)
    return make_agent(**settings)


class TestPrioritisedAgent:
    def test_the_uniform_buffer_is_the_default(self) -> None:
        assert type(make_agent().replay) is NStepReplayBuffer

    def test_the_replay_buffer_is_prioritised_when_asked_for(self) -> None:
        agent = make_per_agent()
        assert isinstance(agent.replay, PrioritisedReplayBuffer)
        assert agent.replay.alpha == pytest.approx(agent.config.per_alpha)

    def test_the_n_step_horizon_is_the_configured_one(self) -> None:
        agent = make_per_agent(n_step=3)
        assert agent.replay.n_step == 3
        assert agent.replay.gamma == pytest.approx(agent.config.gamma)

    def test_learning_rewrites_the_priorities_of_the_sampled_transitions(self) -> None:
        agent = make_per_agent()
        fill(agent)
        before = list(agent.replay.priorities())
        agent.learn()
        after = list(agent.replay.priorities())
        assert after != before

    def test_priorities_track_the_per_sample_kl_loss(self) -> None:
        agent = make_per_agent()
        fill(agent)
        agent.learn()
        priorities = np.asarray(agent.replay.priorities())
        touched = priorities[priorities != 1.0]
        assert touched.size > 0
        assert np.all(touched >= agent.config.per_epsilon)

    def test_beta_anneals_with_the_gradient_steps(self) -> None:
        agent = make_per_agent(per_beta_steps=4)
        fill(agent)
        assert agent.beta() == pytest.approx(agent.config.per_beta_start)
        for _ in range(4):
            agent.learn()
        assert agent.beta() == pytest.approx(agent.config.per_beta_end)

    def test_the_loss_is_importance_weighted(self) -> None:
        agent = make_per_agent()
        fill(agent)
        loss = agent.learn()
        assert loss is not None and np.isfinite(loss)

    def test_learning_still_updates_the_parameters(self) -> None:
        agent = make_per_agent()
        fill(agent)
        before = [p.detach().clone() for p in agent.online_network.parameters()]
        agent.learn()
        after = list(agent.online_network.parameters())
        assert any(not torch.allclose(a, b) for a, b in zip(before, after))

    def test_the_target_network_is_still_resynchronised(self) -> None:
        agent = make_per_agent(target_sync_interval=2)
        fill(agent)
        agent.learn()
        agent.learn()
        agent.online_network.zero_noise()
        agent.target_network.zero_noise()
        for online, target in zip(
            agent.online_network.state_dict().values(),
            agent.target_network.state_dict().values(),
        ):
            assert torch.allclose(online, target)
