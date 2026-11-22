"""The scenario weights have to reach *both* top-down methods, not just the
reported team reward.

``runner.train_joint`` and ``baselines.pmaddpg.PMADDPGMethod`` share the
per-parcel reward (:func:`~chapter4_qolu.env.qolu_rewards.qolu_parcel_rewards`)
and the consensus rule (:mod:`chapter4_qolu.agent.consensus`), so both optional
switches are threaded through both.  These tests check the threading itself -
which weights are handed to which function, in which order - and the invariance
the switches exist to remove.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from chapter4_qolu import runner
from chapter4_qolu.agent.qolu_agent import QOLUAgent, QOLUConfig
from chapter4_qolu.baselines.pmaddpg import PMADDPGMethod
from chapter4_qolu.config import SCENARIOS, QOLURewardConfig
from chapter4_qolu.env.qolu_rewards import qolu_parcel_rewards
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.objectives import TopDownObjective
from chapter4_qolu.runner import train_joint
from chapter4_qolu.tests.baseline_fixtures import make_env

OBJECTIVES: tuple[TopDownObjective, ...] = tuple(TopDownObjective)


def make_stakeholder(env: UrbanLandUseEnv, seed: int = 0) -> QOLUAgent:
    return QOLUAgent(
        QOLUConfig(
            observation_channels=env.observation_channels,
            observation_window=env.config.observation_window,
            conv_channels=(8,),
            hidden_units=32,
            batch_size=8,
            learning_starts=8,
            replay_capacity=200,
            target_sync_interval=5,
            n_step=2,
            epsilon_decay_steps=20,
        ),
        seed=seed,
    )


def make_method(env: UrbanLandUseEnv, **overrides: object) -> PMADDPGMethod:
    settings: dict[str, object] = {
        "batch_size": 16,
        "learning_starts": 16,
        "replay_capacity": 512,
        "conv_channels": (8,),
        "hidden_units": 32,
    }
    settings.update(overrides)
    return PMADDPGMethod.for_env(env, seed=0, **settings)  # type: ignore[arg-type]


def run_joint(
    scenario: int, qolu_reward: QOLURewardConfig, iterations: int = 8
) -> tuple[runner.JointTrainingReport, list[QOLUAgent]]:
    env = make_env(scenario=scenario)
    stakeholders = [make_stakeholder(env, seed=index) for index in range(3)]
    report = train_joint(
        env,
        stakeholders,
        None,
        iterations=iterations,
        seed=17,
        qolu_reward=qolu_reward,
    )
    return report, stakeholders


def parameters_match(left: QOLUAgent, right: QOLUAgent) -> bool:
    return all(
        torch.equal(a, b)
        for a, b in zip(
            left.online_network.parameters(), right.online_network.parameters()
        )
    )


class TestTrainJointWiring:
    def test_the_weights_reach_the_consensus_in_declaration_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []
        original = runner.action_weighted_votes

        def spy(scores: Any, actions: Any, weights: Any = None) -> Any:
            seen.append(weights)
            return original(scores, actions, weights=weights)

        monkeypatch.setattr(runner, "action_weighted_votes", spy)
        env = make_env(scenario=3)
        train_joint(
            env,
            [make_stakeholder(env, seed=index) for index in range(3)],
            None,
            iterations=2,
            seed=0,
            qolu_reward=QOLURewardConfig(weighted_consensus=True),
        )
        assert seen
        for weights in seen:
            assert list(weights) == [
                SCENARIOS[3][objective] for objective in OBJECTIVES
            ]

    def test_no_consensus_weights_are_passed_while_the_switch_is_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []
        original = runner.action_weighted_votes

        def spy(scores: Any, actions: Any, weights: Any = None) -> Any:
            seen.append(weights)
            return original(scores, actions, weights=weights)

        monkeypatch.setattr(runner, "action_weighted_votes", spy)
        env = make_env(scenario=3)
        train_joint(
            env,
            [make_stakeholder(env, seed=index) for index in range(3)],
            None,
            iterations=2,
            seed=0,
        )
        assert seen and all(weights is None for weights in seen)

    def test_the_weights_and_the_stakeholder_count_reach_the_reward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[Any, Any]] = []
        original = runner.qolu_parcel_rewards

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.append((kwargs.get("scenario_weights"), kwargs.get("n_stakeholders")))
            return original(*args, **kwargs)

        monkeypatch.setattr(runner, "qolu_parcel_rewards", spy)
        env = make_env(scenario=2)
        train_joint(
            env,
            [make_stakeholder(env, seed=index) for index in range(3)],
            None,
            iterations=2,
            seed=0,
            qolu_reward=QOLURewardConfig(scale_by_scenario_weight=True),
        )
        assert seen
        for weights, count in seen:
            assert weights == SCENARIOS[2]
            assert count == 3


class TestScenarioInvariance:
    """The scenario invariance the switches can remove, pinned as a test."""

    def test_today_scenarios_one_and_three_train_identical_stakeholders(self) -> None:
        first, agents_one = run_joint(1, QOLURewardConfig())
        second, agents_three = run_joint(3, QOLURewardConfig())
        assert first.n_approved == second.n_approved
        assert first.n_conversions == second.n_conversions
        assert all(
            parameters_match(left, right)
            for left, right in zip(agents_one, agents_three)
        )

    def test_reward_scaling_separates_scenario_three_from_scenario_one(self) -> None:
        config = QOLURewardConfig(scale_by_scenario_weight=True)
        _, agents_one = run_joint(1, config)
        _, agents_three = run_joint(3, config)
        assert not all(
            parameters_match(left, right)
            for left, right in zip(agents_one, agents_three)
        )

    def test_equal_weights_reproduce_todays_run_exactly(self) -> None:
        plain, plain_agents = run_joint(1, QOLURewardConfig())
        switched, switched_agents = run_joint(
            1,
            QOLURewardConfig(scale_by_scenario_weight=True, weighted_consensus=True),
        )
        assert switched.n_approved == plain.n_approved
        assert switched.n_conversions == plain.n_conversions
        assert switched.team_rewards == plain.team_rewards
        assert all(
            parameters_match(left, right)
            for left, right in zip(plain_agents, switched_agents)
        )

    def test_reward_scaling_is_inert_outside_the_per_parcel_reading(self) -> None:
        # Documented scope: ``"team"`` already is ``sum_i w_i R_phi_i +
        # R_society`` and ``"objective"`` is left exactly as published, so the
        # switch must change nothing there rather than weight twice.
        for mode in ("objective", "team"):
            _, plain = run_joint(3, QOLURewardConfig(mode=mode))
            _, scaled = run_joint(
                3, QOLURewardConfig(mode=mode, scale_by_scenario_weight=True)
            )
            assert all(
                parameters_match(left, right) for left, right in zip(plain, scaled)
            )


class TestPMADDPGWiring:
    """The baseline is paid exactly what QOLU is paid."""

    @staticmethod
    def _one_step(
        scenario: int, qolu_reward: QOLURewardConfig, **overrides: object
    ) -> tuple[PMADDPGMethod, Any, Any, Any]:
        env = make_env(scenario=scenario)
        batch = env.reset(seed=0)
        method = make_method(env, qolu_reward=qolu_reward, **overrides)
        method.reset(env, 0)
        grid_before = env.grid
        decision = method.decide(batch)
        result = env.step(decision.decisions)
        method.observe(result)
        return method, env, grid_before, result

    def _assert_matches_qolu(
        self, scenario: int, qolu_reward: QOLURewardConfig
    ) -> None:
        method, env, grid_before, result = self._one_step(scenario, qolu_reward)
        assert method.last_rewards is not None and method.last_actions is not None
        for index, objective in enumerate(OBJECTIVES):
            taken = (
                method.last_actions[:, index] >= method.config.approve_threshold
            ).astype(np.int64)
            expected = qolu_parcel_rewards(
                grid_before,
                result.parcels,
                taken,
                objective,
                proposed=None,
                config=env.config.reward,
                # ``method.qolu_reward`` is the supplied reading with Chapter
                # 3's linearisation constant folded in; that is the reading the
                # agents are paid under, and it is the one QOLU would be paid
                # under if it carried the same constant.
                qolu_reward=method.qolu_reward,
                team_reward=result.team_reward,
                scenario_weights=env.weights,
                n_stakeholders=method.config.n_agents,
            )
            np.testing.assert_allclose(method.last_rewards[:, index], expected)

    def test_scaled_rewards_match_the_qolu_signal_parcel_for_parcel(self) -> None:
        self._assert_matches_qolu(3, QOLURewardConfig(scale_by_scenario_weight=True))

    def test_unscaled_rewards_still_match_the_qolu_signal(self) -> None:
        self._assert_matches_qolu(3, QOLURewardConfig())

    def test_scaling_amplifies_the_woodland_agents_reward(self) -> None:
        # Chapter 3's linearisation constant is switched off here because the
        # shared team term it adds is *deliberately* left unscaled (see
        # ``QOLURewardConfig``), so it would otherwise blunt the ratio this
        # test is about.  ``TestPMADDPGWiring`` above covers it with the
        # constant on.
        plain, *_ = self._one_step(3, QOLURewardConfig(), team_reward_weight=0.0)
        scaled, *_ = self._one_step(
            3, QOLURewardConfig(scale_by_scenario_weight=True), team_reward_weight=0.0
        )
        assert plain.last_rewards is not None and scaled.last_rewards is not None
        np.testing.assert_allclose(
            scaled.last_rewards[:, 1], 1.5 * plain.last_rewards[:, 1]
        )
        np.testing.assert_allclose(
            scaled.last_rewards[:, 0], 0.75 * plain.last_rewards[:, 0]
        )

    def test_equal_weights_leave_the_baseline_reward_untouched(self) -> None:
        plain, *_ = self._one_step(1, QOLURewardConfig())
        scaled, *_ = self._one_step(1, QOLURewardConfig(scale_by_scenario_weight=True))
        assert plain.last_rewards is not None and scaled.last_rewards is not None
        np.testing.assert_array_equal(scaled.last_rewards, plain.last_rewards)

    def test_the_weighted_consensus_can_overturn_the_plain_sum(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two mild approvals against one firm rejection: the published plain sum
        # approves, Scenario 3's weighting hands the guardian the casting vote.
        fixed = (0.7, 0.2, 0.7)

        def decisions(weighted: bool) -> np.ndarray:
            env = make_env(scenario=3)
            batch = env.reset(seed=0)
            method = make_method(
                env,
                # Greedy evaluation mode: the fixed scores below are the point.
                epsilon_greedy=None,
                ou_sigma=0.0,
                qolu_reward=QOLURewardConfig(weighted_consensus=weighted),
            )
            method.reset(env, 0)
            for agent, value in zip(method.agents, fixed):
                monkeypatch.setattr(
                    agent,
                    "approve_scores",
                    lambda tensor, value=value: np.full(
                        int(tensor.shape[0]), value, dtype=np.float64
                    ),
                )
            return method.decide(batch).decisions

        assert decisions(weighted=False).sum() == len(decisions(weighted=False))
        assert decisions(weighted=True).sum() == 0
