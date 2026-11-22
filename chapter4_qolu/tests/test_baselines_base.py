"""Tests for the shared baseline interface (``chapter4_qolu.baselines.base``)."""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.baselines.base import (
    BaselineDecision,
    BaselineMethod,
    BaselineReport,
    conversion_budget,
    run_baseline,
)
from chapter4_qolu.config import EnvConfig
from chapter4_qolu.env.urban_env import APPROVE, REJECT, DecisionBatch, UrbanLandUseEnv
from chapter4_qolu.tests.baseline_fixtures import make_env


class ApproveEverything(BaselineMethod):
    """Minimal conforming method: approve every candidate, learn nothing."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.env: UrbanLandUseEnv | None = None
        self.seeds: list[int | None] = []

    @property
    def name(self) -> str:
        return "approve_everything"

    def reset(self, env: UrbanLandUseEnv, seed: int | None = None) -> None:
        self.calls.append("reset")
        self.seeds.append(seed)
        self.env = env

    def decide(self, batch: DecisionBatch) -> BaselineDecision:
        self.calls.append("decide")
        return BaselineDecision(
            decisions=np.full(len(batch.parcels), APPROVE, dtype=np.int64)
        )

    def observe(self, result) -> None:  # type: ignore[no-untyped-def]
        self.calls.append("observe")

    def train_step(self) -> float | None:
        self.calls.append("train_step")
        return 0.25


class RejectEverything(BaselineMethod):
    """A method that never approves anything and never learns."""

    @property
    def name(self) -> str:
        return "reject_everything"

    def reset(self, env: UrbanLandUseEnv, seed: int | None = None) -> None:
        self._env = env

    def decide(self, batch: DecisionBatch) -> BaselineDecision:
        return BaselineDecision(
            decisions=np.full(len(batch.parcels), REJECT, dtype=np.int64)
        )


class ProposeFirstOnly(BaselineMethod):
    """Approves everything but only proposes the first candidate."""

    @property
    def name(self) -> str:
        return "propose_first_only"

    def reset(self, env: UrbanLandUseEnv, seed: int | None = None) -> None:
        self._env = env

    def decide(self, batch: DecisionBatch) -> BaselineDecision:
        return BaselineDecision(
            decisions=np.full(len(batch.parcels), APPROVE, dtype=np.int64),
            proposals=(batch.parcels[0],),
        )


class BadLength(RejectEverything):
    def decide(self, batch: DecisionBatch) -> BaselineDecision:
        return BaselineDecision(decisions=np.zeros(len(batch.parcels) + 1, np.int64))


class BadAction(RejectEverything):
    def decide(self, batch: DecisionBatch) -> BaselineDecision:
        return BaselineDecision(decisions=np.full(len(batch.parcels), 7, np.int64))


class TestBaselineDecision:
    def test_decisions_are_stored_as_an_integer_array(self) -> None:
        decision = BaselineDecision(decisions=[1, 0, 1])
        assert decision.decisions.dtype == np.int64
        np.testing.assert_array_equal(decision.decisions, [1, 0, 1])

    def test_proposals_default_to_none_meaning_every_candidate(self) -> None:
        assert BaselineDecision(decisions=[0]).proposals is None

    def test_rejects_a_non_one_dimensional_decision_array(self) -> None:
        with pytest.raises(ValueError):
            BaselineDecision(decisions=np.zeros((2, 2), dtype=np.int64))


class TestRunBaseline:
    def test_reports_every_field_the_joint_runner_reports(self) -> None:
        env = make_env()
        report = run_baseline(env, ApproveEverything(), iterations=5, seed=0)
        assert isinstance(report, BaselineReport)
        assert report.method == "approve_everything"
        assert report.iterations == 5
        for series in (
            report.team_rewards,
            report.urban_fraction,
            report.conversion_alignment,
            report.well_being,
            report.n_conversions,
            report.n_proposals,
            report.n_approved,
            report.n_candidates,
        ):
            assert len(series) == 5
        assert set(report.final_metrics) == {
            "land_use",
            "woodland",
            "urban",
            "conversion_alignment",
            "c_align",
            "well_being",
        }

    def test_hooks_are_called_in_the_published_order(self) -> None:
        env = make_env()
        method = ApproveEverything()
        run_baseline(env, method, iterations=2, seed=0)
        assert method.calls == [
            "reset",
            "decide",
            "observe",
            "train_step",
            "decide",
            "observe",
            "train_step",
        ]

    def test_losses_are_collected_when_train_step_returns_one(self) -> None:
        env = make_env()
        report = run_baseline(env, ApproveEverything(), iterations=3, seed=0)
        assert report.losses == (0.25, 0.25, 0.25)

    def test_methods_without_learning_report_no_losses(self) -> None:
        env = make_env()
        report = run_baseline(env, RejectEverything(), iterations=3, seed=0)
        assert report.losses == ()

    def test_the_method_receives_the_environment_on_reset(self) -> None:
        env = make_env()
        method = ApproveEverything()
        run_baseline(env, method, iterations=2, seed=0)
        assert method.env is env

    def test_rejecting_everything_leaves_the_landscape_untouched(self) -> None:
        env = make_env()
        report = run_baseline(env, RejectEverything(), iterations=4, seed=1)
        assert sum(report.n_conversions) == 0
        np.testing.assert_allclose(env.grid.composition, env.initial_grid.composition)

    def test_proposals_restrict_what_may_be_converted(self) -> None:
        env = make_env()
        report = run_baseline(env, ProposeFirstOnly(), iterations=3, seed=2)
        assert all(count <= 1 for count in report.n_conversions)

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        first = run_baseline(make_env(), ApproveEverything(), iterations=6, seed=3)
        second = run_baseline(make_env(), ApproveEverything(), iterations=6, seed=3)
        np.testing.assert_allclose(first.team_rewards, second.team_rewards)

    def test_final_metrics_describe_the_landscape_at_the_end_of_the_run(self) -> None:
        env = make_env(iterations=4)
        report = run_baseline(env, ApproveEverything(), iterations=4, seed=4)
        assert sum(report.n_conversions) > 0
        assert report.urban_fraction[-1] > report.urban_fraction[0]
        assert report.final_metrics["woodland"] <= 1.0

    def test_counts_episodes_when_the_environment_terminates(self) -> None:
        env = make_env(iterations=3)
        report = run_baseline(env, RejectEverything(), iterations=7, seed=5)
        assert report.episodes == 3

    def test_mean_team_reward_matches_the_recorded_series(self) -> None:
        env = make_env()
        report = run_baseline(env, ApproveEverything(), iterations=5, seed=6)
        assert report.mean_team_reward == pytest.approx(np.mean(report.team_rewards))

    def test_rejects_a_non_positive_iteration_count(self) -> None:
        with pytest.raises(ValueError):
            run_baseline(make_env(), ApproveEverything(), iterations=0, seed=0)

    def test_rejects_a_decision_array_of_the_wrong_length(self) -> None:
        with pytest.raises(ValueError):
            run_baseline(make_env(), BadLength(), iterations=1, seed=0)

    def test_rejects_decisions_outside_the_appraisal_action_set(self) -> None:
        with pytest.raises(ValueError):
            run_baseline(make_env(), BadAction(), iterations=1, seed=0)


class TestSharedScaleSettings:
    """Baselines must run at the same scale as QOLU + BUIA, cap included."""

    def test_conversion_budget_reads_the_environment_cap(self) -> None:
        env = make_env(max_conversions_per_iteration=3)
        assert conversion_budget(env) == 3

    def test_a_none_cap_means_every_approved_candidate_converts(self) -> None:
        env = make_env(max_conversions_per_iteration=None, samples_per_iteration=8)
        assert conversion_budget(env) is None
        batch = env.reset(seed=0)
        result = env.step(np.full(len(batch.parcels), APPROVE, dtype=np.int64))
        assert len(result.conversions) == len(set(batch.parcels))

    def test_env_config_accepts_a_none_cap(self) -> None:
        assert EnvConfig(max_conversions_per_iteration=None).iterations == 1000

    def test_env_config_still_rejects_a_negative_cap(self) -> None:
        with pytest.raises(ValueError):
            EnvConfig(max_conversions_per_iteration=-1)

    def test_a_zero_cap_blocks_every_conversion(self) -> None:
        env = make_env(max_conversions_per_iteration=0)
        report = run_baseline(env, ApproveEverything(), iterations=3, seed=7)
        assert sum(report.n_conversions) == 0
