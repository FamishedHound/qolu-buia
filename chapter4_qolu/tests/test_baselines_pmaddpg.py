"""Tests for the P-MADDPG baseline (Chapter 3's property-based MADDPG)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chapter4_qolu.baselines.base import BaselineDecision, run_baseline
from chapter4_qolu.baselines.config import PMADDPGConfig
from chapter4_qolu.baselines.noise import OrnsteinUhlenbeckNoise
from chapter4_qolu.baselines.pmaddpg import (
    PMADDPGMethod,
    appraisal_consensus,
    target_q_value,
)
from chapter4_qolu.baselines.pmaddpg_networks import (
    CentralisedCritic,
    JointReplayBuffer,
    JointTransition,
    ParcelActor,
)
from chapter4_qolu.config import QOLURewardConfig
from chapter4_qolu.env.compliance import parcel_compliance
from chapter4_qolu.env.urban_env import APPROVE, REJECT
from chapter4_qolu.objectives import TopDownObjective
from chapter4_qolu.tests.baseline_fixtures import make_env


def make_method(env, seed: int = 0, **overrides: object) -> PMADDPGMethod:
    settings: dict[str, object] = {
        "batch_size": 16,
        "learning_starts": 16,
        "replay_capacity": 512,
        "conv_channels": (8,),
        "hidden_units": 32,
    }
    settings.update(overrides)
    return PMADDPGMethod.for_env(env, seed=seed, **settings)  # type: ignore[arg-type]


def constant_observation(
    value: float, channels: int = 2, window: int = 3
) -> np.ndarray:
    return np.full((channels, window, window), value, dtype=np.float32)


class TestPMADDPGConfig:
    def test_requires_a_positive_observation_window(self) -> None:
        with pytest.raises(ValueError):
            PMADDPGConfig(observation_channels=4, observation_window=0)

    def test_requires_an_odd_observation_window(self) -> None:
        with pytest.raises(ValueError):
            PMADDPGConfig(observation_channels=4, observation_window=4)

    def test_requires_at_least_one_agent(self) -> None:
        with pytest.raises(ValueError):
            PMADDPGConfig(observation_channels=4, observation_window=3, n_agents=0)

    def test_approve_threshold_must_lie_in_the_unit_interval(self) -> None:
        with pytest.raises(ValueError):
            PMADDPGConfig(
                observation_channels=4, observation_window=3, approve_threshold=1.5
            )

    def test_the_published_threshold_is_one_half(self) -> None:
        assert (
            PMADDPGConfig(
                observation_channels=4, observation_window=3
            ).approve_threshold
            == 0.5
        )

    def test_the_multi_objective_linearisation_constant_is_chapter_threes(self) -> None:
        # alpha = 0.5 (the centre of the published HM sweep) translated into
        # this environment's per-cell mixing form; see PMADDPGConfig.
        config = PMADDPGConfig(observation_channels=4, observation_window=3)
        assert config.team_reward_weight == pytest.approx(0.02)
        # The supplied reward *reading* keeps the published QOLU default; the
        # constant is applied on top of it by the method, for P-MADDPG only.
        assert config.qolu_reward.team_reward_weight == 0.0


class TestOrnsteinUhlenbeckNoise:
    def test_zero_sigma_produces_no_noise(self) -> None:
        noise = OrnsteinUhlenbeckNoise(size=3, sigma=0.0, rng=np.random.default_rng(0))
        np.testing.assert_allclose(noise.sample(), np.zeros(3))

    def test_samples_have_the_requested_shape(self) -> None:
        noise = OrnsteinUhlenbeckNoise(size=5, rng=np.random.default_rng(0))
        assert noise.sample().shape == (5,)

    def test_reset_returns_the_process_to_the_origin(self) -> None:
        noise = OrnsteinUhlenbeckNoise(size=2, rng=np.random.default_rng(0))
        for _ in range(10):
            noise.sample()
        noise.reset()
        np.testing.assert_allclose(noise.state, np.zeros(2))

    def test_the_process_is_temporally_correlated(self) -> None:
        noise = OrnsteinUhlenbeckNoise(
            size=1, sigma=0.5, dt=1.0, rng=np.random.default_rng(1)
        )
        draws = [float(noise.sample()[0]) for _ in range(200)]
        correlation = np.corrcoef(draws[:-1], draws[1:])[0, 1]
        assert correlation > 0.3


class TestNetworks:
    def test_actor_returns_one_approve_score_per_parcel_in_the_unit_interval(
        self,
    ) -> None:
        actor = ParcelActor(
            in_channels=3, window=5, conv_channels=(8,), hidden_units=16
        )
        with torch.no_grad():
            scores = actor(torch.randn(7, 3, 5, 5))
        assert scores.shape == (7, 1)
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0

    def test_critic_maps_a_per_cell_state_and_joint_action_to_one_q_value(self) -> None:
        critic = CentralisedCritic(
            in_channels=3, window=5, n_agents=3, conv_channels=(8,), hidden_units=16
        )
        values = critic(torch.randn(4, 3, 5, 5), torch.rand(4, 3))
        assert values.shape == (4, 1)

    def test_critic_rejects_a_joint_action_with_the_wrong_agent_count(self) -> None:
        critic = CentralisedCritic(
            in_channels=3, window=5, n_agents=3, conv_channels=(8,), hidden_units=16
        )
        with pytest.raises(ValueError):
            critic(torch.randn(4, 3, 5, 5), torch.rand(4, 2))


class TestJointReplayBuffer:
    def _transition(self, value: float) -> JointTransition:
        return JointTransition(
            observation=constant_observation(value),
            actions=np.array([value, value, value], dtype=np.float32),
            rewards=np.array([value, -value, 0.0], dtype=np.float32),
            next_observation=constant_observation(value),
            done=False,
        )

    def test_stores_and_samples_uniform_minibatches(self) -> None:
        buffer = JointReplayBuffer(capacity=10, n_agents=3)
        for index in range(5):
            buffer.push(self._transition(index / 10.0))
        batch = buffer.sample(4, np.random.default_rng(0))
        assert batch.observations.shape == (4, 2, 3, 3)
        assert batch.actions.shape == (4, 3)
        assert batch.rewards.shape == (4, 3)
        assert batch.dones.shape == (4,)

    def test_respects_its_capacity(self) -> None:
        buffer = JointReplayBuffer(capacity=3, n_agents=3)
        for index in range(10):
            buffer.push(self._transition(index / 10.0))
        assert len(buffer) == 3

    def test_rejects_a_transition_with_the_wrong_agent_count(self) -> None:
        buffer = JointReplayBuffer(capacity=3, n_agents=2)
        with pytest.raises(ValueError):
            buffer.push(self._transition(0.5))

    def test_cannot_sample_from_an_empty_buffer(self) -> None:
        with pytest.raises(ValueError):
            JointReplayBuffer(capacity=3, n_agents=3).sample(1, np.random.default_rng())


class TestPublishedCriticTarget:
    """``targetQ <- psi^t(s', upsilon^t(s')) * gamma + r`` (Chapter 3, Alg. 1)."""

    def test_target_is_gamma_times_next_value_plus_reward(self) -> None:
        target = target_q_value(
            next_q=torch.tensor([[2.0], [4.0]]),
            rewards=torch.tensor([1.0, -1.0]),
            gamma=0.5,
        )
        np.testing.assert_allclose(target.squeeze(-1).numpy(), [2.0, 1.0])

    def test_the_published_form_does_not_mask_terminal_transitions(self) -> None:
        target = target_q_value(
            next_q=torch.tensor([[2.0]]),
            rewards=torch.tensor([1.0]),
            gamma=0.5,
            dones=torch.tensor([1.0]),
        )
        assert float(target) == pytest.approx(2.0)

    def test_terminal_masking_is_available_as_the_standard_alternative(self) -> None:
        target = target_q_value(
            next_q=torch.tensor([[2.0]]),
            rewards=torch.tensor([1.0]),
            gamma=0.5,
            dones=torch.tensor([1.0]),
            mask_terminal=True,
        )
        assert float(target) == pytest.approx(1.0)


class TestAppraisalConsensus:
    def test_a_two_thirds_majority_of_strong_approvals_wins(self) -> None:
        decisions = appraisal_consensus(
            [np.array([0.9]), np.array([0.9]), np.array([0.1])], threshold=0.5
        )
        np.testing.assert_array_equal(decisions, [APPROVE])

    def test_two_strong_rejections_outvote_one_weak_approval(self) -> None:
        decisions = appraisal_consensus(
            [np.array([0.6]), np.array([0.1]), np.array([0.1])], threshold=0.5
        )
        np.testing.assert_array_equal(decisions, [REJECT])

    def test_scores_are_thresholded_at_one_half(self) -> None:
        decisions = appraisal_consensus([np.array([0.51, 0.49])], threshold=0.5)
        np.testing.assert_array_equal(decisions, [APPROVE, REJECT])


class TestUpdateDirections:
    """Synthetic per-parcel data: does the published update move the right way?"""

    @staticmethod
    def _method(reward_sign: float) -> PMADDPGMethod:
        env = make_env()
        method = make_method(env, seed=0, mask_terminal=True, batch_size=32)
        method.reset(env, 0)
        rng = np.random.default_rng(0)
        for _ in range(256):
            observation = constant_observation(
                float(rng.random()),
                channels=env.observation_channels,
                window=env.config.observation_window,
            )
            actions = rng.random(3).astype(np.float32)
            reward = reward_sign * np.where(actions > 0.5, 1.0, -1.0)
            method.remember(
                JointTransition(
                    observation=observation,
                    actions=actions,
                    rewards=reward.astype(np.float32),
                    next_observation=observation,
                    done=True,
                )
            )
        for _ in range(200):
            method.train_step()
        return method

    def test_rewarding_approval_pushes_the_actor_score_up(self) -> None:
        method = self._method(reward_sign=1.0)
        scores = method.approve_scores(
            constant_observation(
                0.5, channels=method.config.observation_channels, window=5
            )[None]
        )
        assert float(np.mean(scores)) > 0.5

    def test_punishing_approval_pushes_the_actor_score_down(self) -> None:
        method = self._method(reward_sign=-1.0)
        scores = method.approve_scores(
            constant_observation(
                0.5, channels=method.config.observation_channels, window=5
            )[None]
        )
        assert float(np.mean(scores)) < 0.5

    def test_critic_loss_falls_on_a_fixed_synthetic_batch(self) -> None:
        env = make_env()
        method = make_method(env, seed=1, mask_terminal=True)
        method.reset(env, 1)
        rng = np.random.default_rng(1)
        for _ in range(64):
            observation = constant_observation(
                0.25,
                channels=env.observation_channels,
                window=env.config.observation_window,
            )
            method.remember(
                JointTransition(
                    observation=observation,
                    actions=rng.random(3).astype(np.float32),
                    rewards=np.ones(3, dtype=np.float32),
                    next_observation=observation,
                    done=True,
                )
            )
        first = method.train_step()
        for _ in range(60):
            last = method.train_step()
        assert first is not None and last is not None
        assert last < first

    def test_the_actor_step_raises_the_critic_value_it_maximises(self) -> None:
        env = make_env()
        method = make_method(env, seed=2)
        method.reset(env, 2)
        agent = method.agents[0]
        observations = torch.rand(8, env.observation_channels, 5, 5)
        joint = torch.rand(8, 3)
        before = agent.policy_value(observations, joint, index=0)
        for _ in range(30):
            agent.update_actor(observations, joint, index=0)
        after = agent.policy_value(observations, joint, index=0)
        assert after > before


class TestPerAgentRewards:
    def test_each_agent_is_paid_its_own_objective_compliance(self) -> None:
        env = make_env()
        method = make_method(env)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        grid_before = env.grid
        decision = method.decide(batch)
        result = env.step(decision.decisions, proposals=decision.proposals)
        method.observe(result)

        stored = method.last_rewards
        assert stored is not None
        assert stored.shape == (len(batch.parcels), 3)
        objectives = list(TopDownObjective)
        actions = method.last_actions
        assert actions is not None
        # Chapter 3's linearisation constant adds the same shared term to every
        # cell; the *own-objective* part is what this test is about.
        shared = method.qolu_reward.team_reward_weight * float(result.team_reward)
        for agent_index, objective in enumerate(objectives):
            for parcel_index, parcel in enumerate(batch.parcels):
                compliance = parcel_compliance(
                    grid_before, parcel, objective, env.config.reward
                )
                sign = 1.0 if actions[parcel_index, agent_index] >= 0.5 else -1.0
                expected = sign * (compliance - 0.5) + shared
                assert stored[parcel_index, agent_index] == pytest.approx(expected)

    def test_the_reward_reading_is_configurable(self) -> None:
        env = make_env()
        method = make_method(env, qolu_reward=QOLURewardConfig(mode="team"))
        batch = env.reset(seed=0)
        method.reset(env, 0)
        decision = method.decide(batch)
        result = env.step(decision.decisions, proposals=decision.proposals)
        method.observe(result)
        stored = method.last_rewards
        assert stored is not None
        np.testing.assert_allclose(stored, result.team_reward, atol=1e-6)


class TestMethodBehaviour:
    def test_reports_its_published_name(self) -> None:
        assert make_method(make_env()).name == "p_maddpg"

    def test_one_agent_per_top_down_stakeholder(self) -> None:
        assert len(make_method(make_env()).agents) == 3

    def test_decisions_cover_every_candidate_and_use_the_published_action_set(
        self,
    ) -> None:
        env = make_env()
        method = make_method(env)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        decision = method.decide(batch)
        assert isinstance(decision, BaselineDecision)
        assert decision.decisions.shape == (len(batch.parcels),)
        assert set(np.unique(decision.decisions)) <= {REJECT, APPROVE}

    def test_every_candidate_counts_as_proposed(self) -> None:
        env = make_env()
        method = make_method(env)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        assert method.decide(batch).proposals is None

    def test_exploration_noise_can_be_switched_off(self) -> None:
        """``epsilon = None`` + ``sigma = 0`` is the greedy evaluation mode."""
        env = make_env()
        method = make_method(env, epsilon_greedy=None, ou_sigma=0.0)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        first = method.decide(batch)
        assert method.last_actions is not None
        np.testing.assert_allclose(
            method.last_actions.T, method.approve_scores(batch.observations), rtol=1e-6
        )
        method.reset(env, 0)
        np.testing.assert_array_equal(first.decisions, method.decide(batch).decisions)

    def test_runs_end_to_end_and_learns(self) -> None:
        env = make_env()
        method = make_method(env, learning_starts=32, batch_size=16)
        before = [p.detach().clone() for p in method.agents[0].actor.parameters()]
        report = run_baseline(env, method, iterations=12, seed=0)
        assert report.method == "p_maddpg"
        assert len(report.team_rewards) == 12
        assert report.losses
        assert any(
            not torch.allclose(a, b)
            for a, b in zip(before, method.agents[0].actor.parameters())
        )

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        def run() -> tuple[float, ...]:
            env = make_env()
            return run_baseline(env, make_method(env, seed=5), 8, seed=5).team_rewards

        np.testing.assert_allclose(run(), run())

    def test_honours_the_shared_conversion_budget(self) -> None:
        env = make_env(max_conversions_per_iteration=2)
        report = run_baseline(env, make_method(env), iterations=8, seed=0)
        assert max(report.n_conversions) <= 2
