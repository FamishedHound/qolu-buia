"""P-MADDPG: the published Chapter 3 numbers and the actor-loss regulariser.

Six things are under test here, each either a *published* Chapter 3 number or
a property the implementation has to hold:

1. the Lowe et al. (2017) policy regulariser that stops the actor's sigmoid
   saturating into a range the critic can no longer move it out of;
2. Chapter 3's hard target-network copy "every 250 steps";
3. Chapter 3's epsilon-greedy exploration with ``epsilon = 0.3``;
4. independent per-run and per-agent random streams (a ``seed + index``
   offset would make consecutive run seeds share two of three
   initialisations, and ``torch.manual_seed`` would move the *global*
   generator);
5. per-parcel exploration noise (one size-1 OU process sampled ``batch``
   times is a single per-iteration offset, not per-parcel noise);
6. the variant factory that keeps the unregularised reading reachable.

The broader behavioural tests live in
:mod:`chapter4_qolu.tests.test_baselines_pmaddpg`; this module covers the
hyper-parameters, the regulariser and the random streams.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chapter4_qolu.baselines.base import run_baseline
from chapter4_qolu.baselines.config import PMADDPGConfig
from chapter4_qolu.baselines.pmaddpg import (
    PMADDPG_VARIANTS,
    PMADDPGMethod,
    pmaddpg_reg_only,
    pmaddpg_v1,
    published_pmaddpg,
    variant_settings,
)
from chapter4_qolu.baselines.pmaddpg_networks import JointTransition, ParcelActor
from chapter4_qolu.config import QOLURewardConfig
from chapter4_qolu.tests.baseline_fixtures import make_env

SMALL: dict[str, object] = {
    "batch_size": 16,
    "learning_starts": 16,
    "replay_capacity": 512,
    "conv_channels": (8,),
    "hidden_units": 32,
}


def make_method(env, seed: int = 0, **overrides: object) -> PMADDPGMethod:
    settings: dict[str, object] = dict(SMALL)
    settings.update(overrides)
    return PMADDPGMethod.for_env(env, seed=seed, **settings)  # type: ignore[arg-type]


def constant_observation(value: float, channels: int, window: int) -> np.ndarray:
    return np.full((channels, window, window), value, dtype=np.float32)


def fill_replay(method: PMADDPGMethod, env, reward_sign: float, n: int = 128) -> None:
    """Experience in which approving is unambiguously worth ``reward_sign``."""
    rng = np.random.default_rng(0)
    for _ in range(n):
        observation = constant_observation(
            float(rng.random()),
            channels=env.observation_channels,
            window=env.config.observation_window,
        )
        actions = rng.random(method.config.n_agents).astype(np.float32)
        rewards = reward_sign * np.where(actions >= 0.5, 1.0, -1.0)
        method.remember(
            JointTransition(
                observation=observation,
                actions=actions,
                rewards=rewards.astype(np.float32),
                next_observation=observation,
                done=True,
            )
        )


def actor_logits(method: PMADDPGMethod, observations: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return np.stack(
            [
                agent.actor.logits(observations).reshape(-1).cpu().numpy()
                for agent in method.agents
            ]
        )


# ----------------------------------------------------------------------
# 1. the policy regulariser
# ----------------------------------------------------------------------
class TestPolicyRegulariser:
    """``loss = -mean psi_i(s, upsilon_i(s)) + c * mean(logit^2)``."""

    def test_the_actor_exposes_its_pre_activation_logits(self) -> None:
        actor = ParcelActor(in_channels=2, window=5, conv_channels=(4,), hidden_units=8)
        observations = torch.randn(6, 2, 5, 5)
        with torch.no_grad():
            logits = actor.logits(observations)
            scores = actor(observations)
        assert logits.shape == (6, 1)
        torch.testing.assert_close(torch.sigmoid(logits), scores)

    def test_the_published_default_is_the_reference_implementation_value(self) -> None:
        config = PMADDPGConfig(observation_channels=4, observation_window=3)
        assert config.policy_regulariser == pytest.approx(1e-3)

    def test_a_negative_regulariser_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            PMADDPGConfig(
                observation_channels=4, observation_window=3, policy_regulariser=-1.0
            )

    def test_the_actor_loss_is_minus_q_plus_the_squared_logit_penalty(self) -> None:
        env = make_env()
        method = make_method(env, policy_regulariser=0.25)
        agent = method.agents[0]
        observations = torch.rand(8, env.observation_channels, 5, 5)
        joint = torch.rand(8, 3)
        with torch.no_grad():
            logits = agent.actor.logits(observations)
            substituted = agent._substitute(joint, torch.sigmoid(logits), 0)
            expected = float(
                -agent.critic(observations, substituted).mean()
                + 0.25 * (logits**2).mean()
            )
        assert agent.update_actor(observations, joint, 0) == pytest.approx(
            expected, rel=1e-5
        )

    def test_a_zero_regulariser_reproduces_the_plain_published_loss(self) -> None:
        env = make_env()
        method = make_method(env, policy_regulariser=0.0)
        agent = method.agents[0]
        observations = torch.rand(8, env.observation_channels, 5, 5)
        joint = torch.rand(8, 3)
        with torch.no_grad():
            substituted = agent._substitute(joint, agent.actor(observations), 0)
            expected = float(-agent.critic(observations, substituted).mean())
        assert agent.update_actor(observations, joint, 0) == pytest.approx(
            expected, rel=1e-5
        )

    @staticmethod
    def _saturated_bias_gradient(policy_regulariser: float) -> tuple[float, float]:
        """``(d loss / d output-bias, mean logit)`` deep inside saturation."""
        env = make_env()
        method = make_method(env, policy_regulariser=policy_regulariser)
        agent = method.agents[0]
        with torch.no_grad():
            agent.actor.head[-1].bias.fill_(40.0)
        observations = torch.rand(8, env.observation_channels, 5, 5)
        joint = torch.rand(8, 3)
        with torch.no_grad():
            mean_logit = float(agent.actor.logits(observations).mean())
        agent.update_actor(observations, joint, 0)
        gradient = agent.actor.head[-1].bias.grad
        assert gradient is not None
        return float(gradient.reshape(-1)[0]), mean_logit

    def test_a_saturated_actor_receives_no_gradient_without_the_regulariser(
        self,
    ) -> None:
        # |logit| ~ 40 makes sigma'(z) ~ 4e-18, so the critic can no longer
        # move the actor at all.
        gradient, mean_logit = self._saturated_bias_gradient(0.0)
        assert mean_logit > 39.0
        assert abs(gradient) < 1e-12

    def test_the_regulariser_pulls_a_saturated_actor_back_towards_zero(self) -> None:
        # d/db of c * mean(z^2) = c * mean(2 z): a strictly positive gradient,
        # so the descent step lowers the bias and |logit| with it.
        gradient, mean_logit = self._saturated_bias_gradient(1e-3)
        assert gradient == pytest.approx(1e-3 * 2.0 * mean_logit, rel=1e-4)
        assert gradient > 0.0

    def test_the_regulariser_keeps_the_actor_out_of_saturation_while_training(
        self,
    ) -> None:
        def trained(policy_regulariser: float) -> float:
            env = make_env()
            method = make_method(
                env,
                seed=3,
                policy_regulariser=policy_regulariser,
                actor_learning_rate=1e-3,
                critic_learning_rate=1e-3,
            )
            method.reset(env, 3)
            fill_replay(method, env, reward_sign=1.0)
            for _ in range(200):
                method.train_step()
            probe = torch.rand(16, env.observation_channels, 5, 5)
            return float(np.abs(actor_logits(method, probe)).max())

        assert trained(0.0) > 10.0
        assert trained(1e-3) < 10.0


# ----------------------------------------------------------------------
# 2. Chapter 3's hard target update
# ----------------------------------------------------------------------
def _target_gap(method: PMADDPGMethod) -> float:
    agent = method.agents[0]
    with torch.no_grad():
        return max(
            float((target - online).abs().max())
            for target, online in zip(
                agent.actor_target.parameters(), agent.actor.parameters()
            )
        )


class TestTargetUpdate:
    def test_the_published_interval_is_two_hundred_and_fifty_steps(self) -> None:
        config = PMADDPGConfig(observation_channels=4, observation_window=3)
        assert config.target_update_interval == 250

    def test_a_non_positive_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            PMADDPGConfig(
                observation_channels=4, observation_window=3, target_update_interval=0
            )

    def test_targets_are_frozen_between_hard_copies(self) -> None:
        env = make_env()
        method = make_method(env, target_update_interval=4)
        agent = method.agents[0]
        frozen = [p.detach().clone() for p in agent.actor_target.parameters()]
        observations = torch.rand(8, env.observation_channels, 5, 5)
        joint = torch.rand(8, 3)
        for _ in range(3):
            agent.update_actor(observations, joint, 0)
            agent.update_targets()
        assert all(
            torch.equal(a, b) for a, b in zip(frozen, agent.actor_target.parameters())
        )
        assert _target_gap(method) > 0.0

    def test_the_fourth_step_copies_the_online_networks_exactly(self) -> None:
        env = make_env()
        method = make_method(env, target_update_interval=4)
        agent = method.agents[0]
        observations = torch.rand(8, env.observation_channels, 5, 5)
        joint = torch.rand(8, 3)
        for _ in range(4):
            agent.update_actor(observations, joint, 0)
            agent.update_targets()
        assert _target_gap(method) == pytest.approx(0.0, abs=0.0)

    def test_a_none_interval_keeps_polyak_averaging(self) -> None:
        env = make_env()
        method = make_method(env, target_update_interval=None, tau=0.5)
        agent = method.agents[0]
        before = [p.detach().clone() for p in agent.actor_target.parameters()]
        with torch.no_grad():
            for parameter in agent.actor.parameters():
                parameter.add_(1.0)
        agent.update_targets()
        for old, target, online in zip(
            before, agent.actor_target.parameters(), agent.actor.parameters()
        ):
            torch.testing.assert_close(target, 0.5 * old + 0.5 * online)

    def test_the_target_copy_covers_the_critic_too(self) -> None:
        env = make_env()
        method = make_method(env, target_update_interval=1)
        agent = method.agents[0]
        with torch.no_grad():
            for parameter in agent.critic.parameters():
                parameter.add_(1.0)
        agent.update_targets()
        for target, online in zip(
            agent.critic_target.parameters(), agent.critic.parameters()
        ):
            assert torch.equal(target, online)


# ----------------------------------------------------------------------
# 3. Chapter 3's epsilon-greedy exploration
# ----------------------------------------------------------------------
class TestEpsilonGreedyExploration:
    def test_the_published_epsilon_is_three_tenths(self) -> None:
        config = PMADDPGConfig(observation_channels=4, observation_window=3)
        assert config.epsilon_greedy == pytest.approx(0.3)

    def test_an_epsilon_outside_the_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            PMADDPGConfig(
                observation_channels=4, observation_window=3, epsilon_greedy=1.5
            )

    def test_epsilon_zero_leaves_every_score_untouched(self) -> None:
        env = make_env()
        method = make_method(env, epsilon_greedy=0.0)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        method.decide(batch)
        actions = method.last_actions
        assert actions is not None
        np.testing.assert_allclose(
            actions.T, method.approve_scores(batch.observations), rtol=1e-6
        )

    def test_epsilon_one_replaces_every_score_with_a_uniform_draw(self) -> None:
        env = make_env()
        method = make_method(env, epsilon_greedy=1.0)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        method.decide(batch)
        actions = method.last_actions
        assert actions is not None
        clean = method.approve_scores(batch.observations)
        assert np.all(actions >= 0.0) and np.all(actions <= 1.0)
        assert not np.allclose(actions.T, clean)
        # A uniform draw spreads across [0, 1]; an untrained actor does not.
        assert float(actions.std()) > float(clean.std())

    def test_roughly_epsilon_of_the_scores_are_replaced(self) -> None:
        env = make_env(samples_per_iteration=64)
        method = make_method(env, epsilon_greedy=0.3)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        replaced = 0
        total = 0
        for _ in range(40):
            method.decide(batch)
            actions = method.last_actions
            assert actions is not None
            clean = method.approve_scores(batch.observations).T
            replaced += int(np.count_nonzero(~np.isclose(actions, clean)))
            total += int(actions.size)
        assert 0.2 < replaced / total < 0.4

    def test_epsilon_greedy_is_reproducible_for_a_fixed_seed(self) -> None:
        def draw() -> np.ndarray:
            env = make_env()
            method = make_method(env, seed=11)
            batch = env.reset(seed=11)
            method.reset(env, 11)
            method.decide(batch)
            assert method.last_actions is not None
            return method.last_actions

        np.testing.assert_allclose(draw(), draw())

    def test_setting_epsilon_to_none_selects_the_ornstein_uhlenbeck_alternative(
        self,
    ) -> None:
        env = make_env()
        method = make_method(env, epsilon_greedy=None, ou_sigma=0.0)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        method.decide(batch)
        actions = method.last_actions
        assert actions is not None
        # sigma = 0 is how OU exploration is switched off, so the scores are the
        # actor's own - which epsilon-greedy would still have perturbed.
        np.testing.assert_allclose(
            actions.T, method.approve_scores(batch.observations), rtol=1e-6
        )


# ----------------------------------------------------------------------
# 4. independent random streams
# ----------------------------------------------------------------------
def _initial_parameters(method: PMADDPGMethod) -> list[np.ndarray]:
    return [
        parameter.detach().cpu().numpy().copy()
        for agent in method.agents
        for module in (agent.actor, agent.critic)
        for parameter in module.parameters()
    ]


class TestSeedIndependence:
    def test_run_seeds_zero_and_one_share_no_agent_initialisation(self) -> None:
        env = make_env()
        first = _initial_parameters(make_method(env, seed=0))
        second = _initial_parameters(make_method(env, seed=1))
        assert len(first) == len(second)
        assert not any(
            np.array_equal(a, b) for a in first for b in second if a.shape == b.shape
        )

    def test_consecutive_seeds_do_not_share_whole_agents(self) -> None:
        """``seed + index`` would give run 0 agent 1 == run 1 agent 0."""
        env = make_env()
        runs = [make_method(env, seed=seed) for seed in (0, 1, 2)]
        signatures = [
            [
                float(next(agent.actor.parameters()).detach().sum())
                for agent in method.agents
            ]
            for method in runs
        ]
        flat = [value for row in signatures for value in row]
        assert len(set(flat)) == len(flat)

    def test_the_agents_of_one_run_are_initialised_independently(self) -> None:
        method = make_method(make_env(), seed=0)
        sums = [
            float(next(agent.actor.parameters()).detach().sum())
            for agent in method.agents
        ]
        assert len(set(sums)) == len(sums)

    def test_building_a_method_does_not_move_the_global_torch_generator(self) -> None:
        torch.manual_seed(1234)
        expected = torch.rand(4)
        torch.manual_seed(1234)
        make_method(make_env(), seed=7)
        torch.testing.assert_close(torch.rand(4), expected)

    def test_the_exploration_streams_of_two_runs_are_independent(self) -> None:
        env = make_env()

        def draws(seed: int) -> np.ndarray:
            method = make_method(env, seed=seed, epsilon_greedy=1.0)
            batch = env.reset(seed=0)
            method.reset(env, seed)
            method.decide(batch)
            assert method.last_actions is not None
            return method.last_actions

        assert not np.allclose(draws(0), draws(1))

    def test_a_run_is_still_reproducible_from_its_seed(self) -> None:
        env = make_env()
        first = _initial_parameters(make_method(env, seed=4))
        second = _initial_parameters(make_method(env, seed=4))
        assert all(np.array_equal(a, b) for a, b in zip(first, second))


# ----------------------------------------------------------------------
# 5. per-parcel exploration noise
# ----------------------------------------------------------------------
class TestPerParcelNoise:
    def test_ou_noise_is_drawn_per_parcel_not_once_per_iteration(self) -> None:
        env = make_env(samples_per_iteration=32)
        method = make_method(env, epsilon_greedy=None, ou_sigma=2.0, ou_dt=1.0)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        method.decide(batch)
        actions = method.last_actions
        assert actions is not None
        clean = method.approve_scores(batch.observations).T
        perturbation = actions - clean
        # One scalar OU process reused for every parcel makes this exactly zero.
        assert float(perturbation[:, 0].std()) > 1e-6

    def test_the_noise_process_matches_the_decision_buffer_width(self) -> None:
        env = make_env(samples_per_iteration=16)
        method = make_method(env, epsilon_greedy=None, ou_sigma=0.2)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        method.decide(batch)
        assert all(noise.size == len(batch.parcels) for noise in method.noise)

    def test_the_noise_process_is_temporally_correlated_per_parcel(self) -> None:
        env = make_env(samples_per_iteration=32)
        method = make_method(env, epsilon_greedy=None, ou_sigma=1.0, ou_dt=1.0)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        method.decide(batch)  # sizes the processes to the decision buffer
        first = method.noise[0].sample()
        second = method.noise[0].sample()
        assert np.corrcoef(first, second)[0, 1] > 0.3


# ----------------------------------------------------------------------
# 6. the variant factory
# ----------------------------------------------------------------------
class TestVariantFactory:
    def test_the_table_carries_exactly_the_three_readings(self) -> None:
        assert set(PMADDPG_VARIANTS) == {"published", "v1", "reg_only"}

    def test_the_published_variant_changes_nothing(self) -> None:
        assert variant_settings("published") == {}

    def test_an_unknown_variant_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            variant_settings("not_a_variant")

    def test_for_env_rejects_an_unknown_variant(self) -> None:
        with pytest.raises(ValueError):
            PMADDPGMethod.for_env(make_env(), seed=0, variant="nope")

    def test_v1_restores_the_first_reconstruction(self) -> None:
        config = make_method(make_env(), variant="v1").config
        assert config.actor_learning_rate == pytest.approx(1e-3)
        assert config.critic_learning_rate == pytest.approx(1e-3)
        assert config.gamma == pytest.approx(0.99)
        assert config.tau == pytest.approx(0.01)
        assert config.target_update_interval is None
        assert config.epsilon_greedy is None
        assert config.policy_regulariser == pytest.approx(0.0)
        assert config.team_reward_weight == pytest.approx(0.0)

    def test_reg_only_is_v1_plus_the_regulariser(self) -> None:
        v1 = dict(variant_settings("v1"))
        reg = dict(variant_settings("reg_only"))
        assert reg.pop("policy_regulariser") == pytest.approx(1e-3)
        v1.pop("policy_regulariser")
        assert reg == v1

    def test_the_published_variant_is_the_config_default(self) -> None:
        config = make_method(make_env(), variant="published").config
        assert config.actor_learning_rate == pytest.approx(1e-4)
        assert config.critic_learning_rate == pytest.approx(1e-4)
        assert config.gamma == pytest.approx(0.1)
        assert config.target_update_interval == 250
        assert config.epsilon_greedy == pytest.approx(0.3)
        assert config.policy_regulariser == pytest.approx(1e-3)

    def test_the_factory_functions_mirror_the_table(self) -> None:
        env = make_env()
        pairs = (
            (published_pmaddpg, "published"),
            (pmaddpg_v1, "v1"),
            (pmaddpg_reg_only, "reg_only"),
        )
        for factory, variant in pairs:
            built = factory(env, seed=0, **SMALL)  # type: ignore[arg-type]
            expected = PMADDPGMethod.for_env(
                env, seed=0, variant=variant, **SMALL  # type: ignore[arg-type]
            )
            assert built.config == expected.config

    def test_explicit_overrides_win_over_the_variant(self) -> None:
        method = make_method(make_env(), variant="v1", gamma=0.5)
        assert method.config.gamma == pytest.approx(0.5)

    def test_every_variant_runs_end_to_end(self) -> None:
        for variant in sorted(PMADDPG_VARIANTS):
            env = make_env()
            method = make_method(env, variant=variant, learning_starts=32)
            report = run_baseline(env, method, iterations=8, seed=0)
            assert report.method == "p_maddpg"
            assert len(report.team_rewards) == 8


# ----------------------------------------------------------------------
# Chapter 3's published hyper-parameters
# ----------------------------------------------------------------------
class TestPublishedHyperParameters:
    """Chapter 3 lines 392-393 and the reward of line 139."""

    def test_the_published_learning_rate_is_one_ten_thousandth(self) -> None:
        config = PMADDPGConfig(observation_channels=4, observation_window=3)
        assert config.actor_learning_rate == pytest.approx(1e-4)
        assert config.critic_learning_rate == pytest.approx(1e-4)

    def test_the_published_discount_is_one_tenth(self) -> None:
        config = PMADDPGConfig(observation_channels=4, observation_window=3)
        assert config.gamma == pytest.approx(0.1)

    def test_the_linearisation_constant_reaches_the_per_parcel_reward(self) -> None:
        env = make_env()
        method = make_method(env, team_reward_weight=0.5)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        decision = method.decide(batch)
        result = env.step(decision.decisions, proposals=decision.proposals)
        method.observe(result)

        plain = make_method(env, team_reward_weight=0.0)
        plain.reset(env, 0)
        plain._grid_before = method._grid_before
        plain._batch = method._batch
        plain.last_actions = method.last_actions
        plain.observe(result)

        assert method.last_rewards is not None and plain.last_rewards is not None
        np.testing.assert_allclose(
            method.last_rewards - plain.last_rewards,
            0.5 * float(result.team_reward),
            atol=1e-6,
        )

    def test_the_campaign_reward_reading_does_not_erase_the_constant(self) -> None:
        """``scenario_worker`` passes its own ``QOLURewardConfig``."""
        method = make_method(
            make_env(),
            qolu_reward=QOLURewardConfig(mode="per_parcel"),
            team_reward_weight=0.25,
        )
        assert method.qolu_reward.team_reward_weight == pytest.approx(0.25)
        assert method.qolu_reward.mode == "per_parcel"

    def test_a_none_constant_leaves_the_supplied_reward_config_alone(self) -> None:
        method = make_method(
            make_env(),
            qolu_reward=QOLURewardConfig(team_reward_weight=0.75),
            team_reward_weight=None,
        )
        assert method.qolu_reward.team_reward_weight == pytest.approx(0.75)

    def test_a_negative_constant_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            PMADDPGConfig(
                observation_channels=4, observation_window=3, team_reward_weight=-0.1
            )
