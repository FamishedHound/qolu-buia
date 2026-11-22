"""Whether the learnt policy separates the candidates it is shown.

The approval rate answers "how often did this stakeholder say yes", and a
constant policy and a discriminating one can both answer it with the same
number: a stakeholder that approves every candidate because ``Q(s, approve)``
exceeds ``Q(s, reject)`` by the same amount everywhere reports an approval rate
of 1.0, and so does one that approves 100 candidates for 100 different reasons
and would have refused the 101st.

The statistic that tells those two apart is the *spread* of the quality score
``Q(s, approve) - Q(s, reject)`` across the candidates appraised.  A policy that
separates nothing has a spread at or near zero; a state-dependent one does not.
The score is computed once per appraisal during the rollout - it is what the
consensus sums.  These tests pin that it is logged, summarised and reported.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.agent.buia_agent import BUIAAgent, BUIAConfig
from chapter4_qolu.agent.qolu_agent import QOLUAgent, QOLUConfig
from chapter4_qolu.config import SCENARIOS, EnvConfig
from chapter4_qolu.diagnostics.collect import collect_appraisals
from chapter4_qolu.diagnostics.records import AppraisalLog, DiagnosticsLog
from chapter4_qolu.diagnostics.report import render_diagnostics_report
from chapter4_qolu.diagnostics.summary import diagnostics_summary, policy_record
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.run_distribution_diagnostics import (
    DiagnosticsSettings,
    diagnostics_paths,
    run_distribution_diagnostics,
)
from chapter4_qolu.tests.raster_fixtures import write_geotiff

ITERATIONS = 4
SAMPLES = 6


def make_grid(size: int = 10) -> ParcelGrid:
    composition = np.zeros((N_AGGREGATE_CLASSES, size, size), dtype=np.float32)
    composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 1.0
    composition[:, 0, 0] = 0.0
    composition[band_index(AggregateClass.BUILT_UP), 0, 0] = 1.0
    return ParcelGrid(composition)


def make_env() -> UrbanLandUseEnv:
    return UrbanLandUseEnv(
        make_grid(),
        config=EnvConfig(
            grid_size=10,
            samples_per_iteration=SAMPLES,
            max_conversions_per_iteration=3,
            iterations=1000,
            observation_window=5,
            buia_observation_window=5,
        ),
        weights=SCENARIOS[1],
    )


def make_stakeholder(env: UrbanLandUseEnv, seed: int = 0) -> QOLUAgent:
    return QOLUAgent(
        QOLUConfig(
            observation_channels=env.observation_channels,
            observation_window=env.config.observation_window,
            batch_size=8,
            learning_starts=8,
            replay_capacity=200,
            n_step=2,
            n_atoms=11,
            v_min=-5.0,
            v_max=5.0,
        ),
        seed=seed,
    )


def make_investor(env: UrbanLandUseEnv) -> BUIAAgent:
    return BUIAAgent(
        BUIAConfig(
            observation_channels=env.buia_observation_channels,
            observation_window=env.config.buia_observation_window,
            max_proposals=3,
        ),
        seed=0,
    )


def collect(n_stakeholders: int = 2) -> DiagnosticsLog:
    env = make_env()
    stakeholders = [
        make_stakeholder(env, seed=index) for index in range(n_stakeholders)
    ]
    return collect_appraisals(
        env,
        stakeholders,
        make_investor(env),
        iterations=ITERATIONS,
        seed=0,
    )


def make_log(
    quality: np.ndarray,
    *,
    stakeholder: int = 0,
    objective: str = "land_use",
    iterations: np.ndarray | None = None,
) -> AppraisalLog:
    """An appraisal log whose only interesting column is the quality score."""
    size = int(np.asarray(quality).size)
    rng = np.random.default_rng(7)
    atoms = np.linspace(-5.0, 5.0, 11)
    probabilities = rng.random((size, 11))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return AppraisalLog(
        stakeholder=stakeholder,
        objective=objective,
        atoms=atoms,
        probabilities=probabilities,
        reference_probabilities=probabilities.copy(),
        actions=(np.asarray(quality) > 0).astype(np.int64),
        quality_scores=np.asarray(quality, dtype=np.float64),
        rewards=np.zeros(size),
        bootstrap_values=np.zeros(size),
        iteration_index=(
            np.zeros(size, dtype=np.int64) if iterations is None else iterations
        ),
        gamma=0.99,
        n_step=2,
    )


class TestTheQualityScoreIsLogged:
    def test_one_score_per_appraisal(self) -> None:
        log = collect()
        for entry in log.logs:
            assert entry.quality_scores.shape == (ITERATIONS * SAMPLES,)
            assert np.isfinite(entry.quality_scores).all()

    def test_the_greedy_action_is_the_sign_of_the_logged_score(self) -> None:
        """Approve is ``argmax_a E[Z]``, which is exactly ``quality > 0``.

        So the two columns are not independent, and their agreement pins that
        the logged score really is the one the decision was taken on rather
        than a second, differently-timed evaluation of the network.
        """
        log = collect()
        for entry in log.logs:
            np.testing.assert_array_equal(
                entry.actions, (entry.quality_scores > 0).astype(entry.actions.dtype)
            )

    def test_a_mismatched_column_is_refused(self) -> None:
        with pytest.raises(ValueError, match="quality_scores"):
            AppraisalLog(
                stakeholder=0,
                objective="land_use",
                atoms=np.linspace(-5.0, 5.0, 11),
                probabilities=np.full((5, 11), 1.0 / 11.0),
                reference_probabilities=np.full((5, 11), 1.0 / 11.0),
                actions=np.zeros(5, dtype=np.int64),
                quality_scores=np.zeros(4),
                rewards=np.zeros(5),
                bootstrap_values=np.zeros(5),
                iteration_index=np.zeros(5, dtype=np.int64),
                gamma=0.99,
                n_step=2,
            )

    def test_the_pooled_log_concatenates_them(self) -> None:
        log = collect(n_stakeholders=3)
        np.testing.assert_allclose(
            log.pooled().quality_scores,
            np.concatenate([entry.quality_scores for entry in log.logs]),
        )


class TestTheRecordCarriesTheSpread:
    def test_a_constant_policy_has_no_spread(self) -> None:
        record = make_log(np.full(20, 3.5)).as_record()["quality_score"]
        assert record["sd"] == pytest.approx(0.0)
        assert record["mean"] == pytest.approx(3.5)
        assert record["mean_absolute"] == pytest.approx(3.5)
        assert record["no_spread"] is True

    def test_a_discriminating_policy_does(self) -> None:
        record = make_log(np.linspace(-2.0, 2.0, 20)).as_record()["quality_score"]
        assert record["sd"] > 0.1
        assert record["min"] == pytest.approx(-2.0)
        assert record["max"] == pytest.approx(2.0)
        assert record["no_spread"] is False

    def test_the_within_iteration_spread_is_reported_separately(self) -> None:
        """A score constant *within* an iteration separates no candidate.

        The landscape moves between iterations, so a policy can vary across a
        rollout while ranking every candidate of a single decision buffer
        identically - which is the thing the consensus actually needs.
        """
        iterations = np.repeat(np.arange(4), 5)
        constant = make_log(
            np.repeat(np.array([1.0, 2.0, 3.0, 4.0]), 5), iterations=iterations
        ).as_record()["quality_score"]
        assert constant["sd"] > 0.0
        assert constant["within_iteration_sd"] == pytest.approx(0.0)

        varying = make_log(
            np.arange(20, dtype=np.float64), iterations=iterations
        ).as_record()["quality_score"]
        assert varying["within_iteration_sd"] > 0.0

    def test_the_fraction_in_favour_agrees_with_the_approval_rate(self) -> None:
        record = make_log(np.linspace(-1.0, 3.0, 21)).as_record()
        assert record["quality_score"]["fraction_positive"] == pytest.approx(
            record["approval_rate"]
        )

    def test_the_spread_ratio_is_relative_to_the_typical_magnitude(self) -> None:
        record = make_log(np.full(20, 100.0) + 1e-9).as_record()["quality_score"]
        assert record["spread_ratio"] == pytest.approx(0.0, abs=1e-9)
        assert record["no_spread"] is True


class TestThePolicyBlockOfTheSummary:
    @staticmethod
    def _summary(quality: list[np.ndarray]) -> dict:
        log = DiagnosticsLog(
            logs=tuple(
                make_log(values, stakeholder=index, objective=name)
                for index, (values, name) in enumerate(
                    zip(quality, ("land_use", "woodland", "urban"))
                )
            ),
            iterations=1,
            samples_per_iteration=len(quality[0]),
            qolu_reward_mode="per_parcel",
            scenario_weights=(0.33, 0.33, 0.33),
        )
        return diagnostics_summary(log, np.random.default_rng(0))

    def test_every_stakeholder_is_named_with_its_rate_and_its_spread(self) -> None:
        summary = self._summary([np.linspace(-1.0, 1.0, 20), np.full(20, 2.0)])
        policy = summary["policy"]
        assert [entry["stakeholder"] for entry in policy["stakeholders"]] == [0, 1]
        assert policy["stakeholders"][0]["objective"] == "land_use"
        assert policy["stakeholders"][0]["quality_score"]["sd"] > 0.0
        assert policy["stakeholders"][1]["quality_score"]["sd"] == pytest.approx(0.0)
        assert policy["stakeholders"][1]["approval_rate"] == pytest.approx(1.0)

    def test_a_constant_policy_is_reported_as_not_state_dependent(self) -> None:
        summary = self._summary([np.full(20, 2.0), np.full(20, -2.0)])
        policy = summary["policy"]
        assert policy["n_stakeholders"] == 2
        assert policy["n_discriminating"] == 0
        assert policy["n_degenerate_approval"] == 2
        assert policy["state_dependent"] is False
        assert "no candidate" in policy["statement"]

    def test_a_discriminating_policy_is_reported_as_state_dependent(self) -> None:
        summary = self._summary(
            [np.linspace(-1.0, 1.0, 20), np.linspace(-3.0, 1.0, 20)]
        )
        policy = summary["policy"]
        assert policy["n_discriminating"] == 2
        assert policy["n_degenerate_approval"] == 0
        assert policy["state_dependent"] is True

    def test_the_helper_is_callable_on_a_log_directly(self) -> None:
        log = DiagnosticsLog(
            logs=(make_log(np.linspace(-1.0, 1.0, 20)),),
            iterations=1,
            samples_per_iteration=20,
            qolu_reward_mode="per_parcel",
            scenario_weights=(0.33, 0.33, 0.33),
        )
        assert policy_record(log)["n_stakeholders"] == 1


class TestTheReportPrintsIt:
    def test_the_section_carries_the_rate_and_the_spread(self) -> None:
        log = DiagnosticsLog(
            logs=(make_log(np.full(20, 2.0)),),
            iterations=1,
            samples_per_iteration=20,
            qolu_reward_mode="team",
            scenario_weights=(0.33, 0.33, 0.33),
        )
        summary = diagnostics_summary(log, np.random.default_rng(0))
        report = render_diagnostics_report(summary)
        assert "Policy state-dependence" in report
        assert "SD of the quality score" in report
        assert summary["policy"]["statement"] in report

    def test_a_summary_written_before_the_block_existed_still_renders(self) -> None:
        log = DiagnosticsLog(
            logs=(make_log(np.full(20, 2.0)),),
            iterations=1,
            samples_per_iteration=20,
            qolu_reward_mode="team",
            scenario_weights=(0.33, 0.33, 0.33),
        )
        summary = diagnostics_summary(log, np.random.default_rng(0))
        summary.pop("policy")
        report = render_diagnostics_report(summary)
        assert "Policy state-dependence" not in report
        assert "## 1. How the distribution is used" in report


class TestTheSweepCellCarriesIt:
    def test_a_cell_row_keeps_each_stakeholders_spread(self, tmp_path: Path) -> None:
        """A sweep row is read off the cell's JSON, so it has to carry it too."""
        from chapter4_qolu.diagnostics.sweep_summary import cell_record

        log = DiagnosticsLog(
            logs=(make_log(np.linspace(-1.0, 1.0, 20)),),
            iterations=1,
            samples_per_iteration=20,
            qolu_reward_mode="per_parcel",
            scenario_weights=(0.33, 0.33, 0.33),
        )
        summary = diagnostics_summary(log, np.random.default_rng(0))
        settings = DiagnosticsSettings(
            raster=Path("(none)"), tag="cell", output_dir=tmp_path
        )
        paths = diagnostics_paths(settings)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.summary.write_text(json.dumps(summary), encoding="utf-8")

        record = cell_record(settings)
        assert record is not None
        assert record["policy"]["n_stakeholders"] == 1
        assert record["stakeholders"][0]["quality_score"]["sd"] > 0.0


@pytest.fixture()
def tiny_class_raster(tmp_path: Path) -> Path:
    rng = np.random.default_rng(20260929)
    codes = rng.integers(1, 11, size=(24, 24), dtype=np.uint8)
    return write_geotiff(
        tmp_path / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        codes,
        origin_e_m=400_000.0,
        origin_n_m=124_000.0,
    )


@pytest.fixture()
def tiny_settings(tiny_class_raster: Path, tmp_path: Path) -> DiagnosticsSettings:
    return DiagnosticsSettings(
        raster=tiny_class_raster,
        tag="tiny",
        window_label="E402_N110_6x6",
        scale=ScaleConfig(grid_size=6, conversion_fraction=1.0, max_conversions=1),
        scenario=1,
        seed=0,
        stakeholders=2,
        iterations=30,
        eval_iterations=3,
        qolu_reward_mode="per_parcel",
        output_dir=tmp_path / "results",
    )


class TestEndToEnd:
    def test_the_json_and_the_report_both_carry_the_spread(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        summary = run_distribution_diagnostics(replace(tiny_settings))
        stored = json.loads(
            diagnostics_paths(tiny_settings).summary.read_text(encoding="utf-8")
        )
        assert stored["policy"] == summary["policy"]
        assert len(stored["policy"]["stakeholders"]) == 2
        for entry in stored["policy"]["stakeholders"]:
            assert set(entry["quality_score"]) >= {
                "mean",
                "sd",
                "min",
                "max",
                "mean_absolute",
                "within_iteration_sd",
                "spread_ratio",
                "fraction_positive",
                "no_spread",
            }
        report = diagnostics_paths(tiny_settings).report.read_text(encoding="utf-8")
        assert "Policy state-dependence" in report

    def test_the_per_stakeholder_record_carries_it_too(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        summary = run_distribution_diagnostics(tiny_settings)
        for entry in summary["rollout"]["stakeholders"]:
            assert "quality_score" in entry
            assert np.isfinite(entry["quality_score"]["sd"])
