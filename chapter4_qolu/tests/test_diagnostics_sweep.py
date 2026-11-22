"""The scenario x seed sweep over the return-distribution diagnostics.

One diagnostics run is one seed of one scenario, and a single seed cannot say
whether a CRPS skill score is a property of the method or of the seed.  These
tests pin the sweep that runs the cells, the aggregation across them and the
document the chapter quotes.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.diagnostics.plan import diagnostics_paths
from chapter4_qolu.diagnostics.summary import CALIBRATION_CAVEAT
from chapter4_qolu.diagnostics.sweep import (
    DEFAULT_SWEEP_SCENARIOS,
    DEFAULT_SWEEP_SEEDS,
    cell_settings,
    run_sweep,
    sweep_cells,
    sweep_paths,
)
from chapter4_qolu.diagnostics.sweep_report import render_sweep_report
from chapter4_qolu.diagnostics.sweep_summary import (
    aggregate_records,
    cell_record,
    flatten_target,
)
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.run_distribution_diagnostics import DiagnosticsSettings
from chapter4_qolu.tests.raster_fixtures import write_geotiff


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
        stakeholders=1,
        iterations=2,
        eval_iterations=2,
        output_dir=tmp_path / "results",
    )


class TestSweepLayout:
    def test_one_cell_per_scenario_and_seed(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        cells = sweep_cells(tiny_settings, scenarios=(1, 2), seeds=(0, 1))
        assert len(cells) == 4
        assert {(cell.scenario, cell.seed) for cell in cells} == {
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
        }

    def test_each_cell_gets_its_own_tag_and_folder(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        cell = cell_settings(tiny_settings, scenario=3, seed=2)
        assert cell.tag == "tiny_s3_seed2"
        root = diagnostics_paths(cell).root
        assert root.parent == sweep_paths(tiny_settings).runs
        assert root.name == "distribution_diagnostics_tiny_s3_seed2"

    def test_the_sweep_writes_beside_its_runs(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        paths = sweep_paths(tiny_settings)
        assert paths.runs.parent == paths.root
        assert paths.summary.parent == paths.root
        assert paths.summary.name.endswith(".json")
        assert paths.report.name.endswith(".md")

    def test_a_cell_keeps_everything_but_the_scenario_seed_and_tag(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        cell = cell_settings(tiny_settings, scenario=4, seed=1)
        assert cell.qolu_reward_mode == tiny_settings.qolu_reward_mode
        assert cell.window_label == tiny_settings.window_label
        assert cell.iterations == tiny_settings.iterations
        assert cell.method == tiny_settings.method

    def test_the_published_defaults_are_four_scenarios_and_three_seeds(self) -> None:
        assert DEFAULT_SWEEP_SCENARIOS == (1, 2, 3, 4)
        assert len(DEFAULT_SWEEP_SEEDS) >= 3


class TestFlatten:
    def test_a_target_block_flattens_to_the_reported_scalars(self) -> None:
        target = {
            "n": 10,
            "mae": 1.0,
            "rmse": 2.0,
            "bias": -0.5,
            "mean_predicted": 0.25,
            "mean_realised": 0.75,
            "sd_realised": 3.0,
            "pit": {
                "ks_statistic": 0.1,
                "ks_p_value": 0.4,
                "out_of_support_fraction": 0.0,
            },
            "coverage": [
                {"level": 0.5, "empirical": 0.48, "mean_width": 12.0},
                {"level": 0.9, "empirical": 0.91, "mean_width": 30.0},
            ],
            "crps": {
                "model": 1.5,
                "reference": {"initialisation": 3.0, "climatology": 2.0},
                "skill": {"initialisation": 0.5, "climatology": 0.25},
            },
            "boundary_mass": {"total_mean": 0.01, "flagged": False},
            "support": {
                "delta_z": 8.0,
                "realised_sd_in_atoms": 0.4,
                "distinct_atoms_used": 3,
                "resolves_the_signal": False,
            },
        }
        flat = flatten_target(target)
        assert flat["mae"] == 1.0
        assert flat["ks_statistic"] == 0.1
        assert flat["coverage_50"] == 0.48
        assert flat["width_90"] == 30.0
        assert flat["skill_climatology"] == 0.25
        assert flat["skill_initialisation"] == 0.5
        assert flat["boundary_mass"] == 0.01
        assert flat["resolves_the_signal"] is False
        assert flat["n"] == 10


class TestAggregation:
    def test_mean_sd_and_se_across_cells(self) -> None:
        records = [
            {"targets": {"tail_return": {"skill_climatology": value}}}
            for value in (0.0, 1.0, 2.0)
        ]
        aggregate = aggregate_records(records, target="tail_return")
        entry = aggregate["skill_climatology"]
        assert entry["n"] == 3
        assert entry["mean"] == pytest.approx(1.0)
        assert entry["sd"] == pytest.approx(1.0)
        assert entry["se"] == pytest.approx(1.0 / np.sqrt(3))
        assert entry["values"] == [0.0, 1.0, 2.0]

    def test_an_empty_set_of_cells_aggregates_to_nothing(self) -> None:
        assert aggregate_records([], target="tail_return") == {}


class TestDryRun:
    def test_a_dry_run_writes_nothing_and_counts_the_cells(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        plan = run_sweep(
            tiny_settings, scenarios=(1, 2), seeds=(0, 1), parallel=1, dry_run=True
        )
        paths = sweep_paths(tiny_settings)
        assert not paths.summary.exists()
        assert not paths.report.exists()
        assert plan["dry_run"] is True
        assert plan["plan"]["n_cells"] == 4
        assert "estimated_serial" in plan["plan"]


class TestEndToEnd:
    @pytest.fixture()
    def finished(self, tiny_settings: DiagnosticsSettings) -> dict:
        return run_sweep(tiny_settings, scenarios=(1, 2), seeds=(0, 1), parallel=1)

    def test_it_writes_a_sweep_summary_and_report(
        self, tiny_settings: DiagnosticsSettings, finished: dict
    ) -> None:
        paths = sweep_paths(tiny_settings)
        assert paths.summary.is_file()
        assert paths.report.is_file()
        assert finished["n_cells"] == 4
        assert finished["n_complete"] == 4

    def test_every_cell_keeps_its_own_documents(
        self, tiny_settings: DiagnosticsSettings, finished: dict
    ) -> None:
        for cell in sweep_cells(tiny_settings, scenarios=(1, 2), seeds=(0, 1)):
            assert diagnostics_paths(cell).summary.is_file()
            assert diagnostics_paths(cell).report.is_file()

    def test_the_cell_rows_carry_the_headline_statistics(
        self, tiny_settings: DiagnosticsSettings, finished: dict
    ) -> None:
        row = finished["cells"][0]
        assert {"scenario", "seed", "tag", "targets", "stakeholders"} <= set(row)
        flat = row["targets"]["tail_return"]
        for key in (
            "mae",
            "rmse",
            "ks_statistic",
            "coverage_50",
            "coverage_90",
            "skill_climatology",
            "skill_initialisation",
            "boundary_mass",
        ):
            assert key in flat

    def test_it_aggregates_overall_per_scenario_and_per_objective(
        self, finished: dict
    ) -> None:
        assert finished["overall"]["skill_climatology"]["n"] == 4
        assert set(finished["per_scenario"]) == {"1", "2"}
        assert finished["per_scenario"]["1"]["skill_climatology"]["n"] == 2
        assert finished["per_objective"]
        for entry in finished["per_objective"].values():
            assert entry["skill_climatology"]["n"] == 4

    def test_the_verdict_counts_the_cells_rather_than_asserting_one(
        self, finished: dict
    ) -> None:
        verdict = finished["verdict"]
        assert set(verdict) >= {
            "n_cells",
            "n_state_conditional",
            "n_calibrated",
            "n_support_adequate",
            "statement",
        }
        assert verdict["n_cells"] == 4
        assert 0 <= verdict["n_state_conditional"] <= 4
        assert isinstance(verdict["statement"], str) and verdict["statement"]

    def test_the_sweep_repeats_the_calibration_caveat(
        self, tiny_settings: DiagnosticsSettings, finished: dict
    ) -> None:
        assert finished["verdict"]["calibration_caveat"] == CALIBRATION_CAVEAT
        report = sweep_paths(tiny_settings).report.read_text(encoding="utf-8")
        assert CALIBRATION_CAVEAT.split(".")[0] in report

    def test_the_cell_rows_carry_the_appraisal_count(self, finished: dict) -> None:
        for cell in finished["cells"]:
            assert cell["targets"]["tail_return"]["n"] > 0

    def test_all_three_targets_are_aggregated_side_by_side(
        self, tiny_settings: DiagnosticsSettings, finished: dict
    ) -> None:
        """The headline is one reading; the other two are never dropped.

        ``n_step_target`` is what the network is literally regressed onto, so a
        sweep that only aggregated the headline would hide the one column that
        says whether the model fits its own training target.
        """
        assert set(finished["by_target"]) == {
            "n_step_reward_sum",
            "n_step_target",
            "tail_return",
        }
        for aggregate in finished["by_target"].values():
            assert aggregate["skill_climatology"]["n"] == 4
        assert finished["by_target"][finished["headline_target"]] == finished["overall"]
        report = sweep_paths(tiny_settings).report.read_text(encoding="utf-8")
        assert "n_step_target" in report

    def test_the_report_names_the_aggregate_number(
        self, tiny_settings: DiagnosticsSettings, finished: dict
    ) -> None:
        report = sweep_paths(tiny_settings).report.read_text(encoding="utf-8")
        assert "CRPS skill" in report
        assert "scenario" in report.lower()
        assert "of 4" in report or "4 of" in report

    def test_the_rendered_report_reads_from_the_summary_alone(
        self, finished: dict
    ) -> None:
        assert render_sweep_report(finished) == render_sweep_report(finished)
        assert "# QOLU return-distribution diagnostics" in render_sweep_report(finished)


class TestCellRecord:
    def test_an_unfinished_cell_reads_as_none(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        assert cell_record(cell_settings(tiny_settings, 1, 0)) is None

    def test_a_finished_cell_reads_back_its_own_numbers(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        run_sweep(tiny_settings, scenarios=(1,), seeds=(0,), parallel=1)
        cell = cell_settings(tiny_settings, 1, 0)
        record = cell_record(cell)
        assert record is not None
        summary = json.loads(diagnostics_paths(cell).summary.read_text("utf-8"))
        assert record["targets"]["tail_return"]["mae"] == pytest.approx(
            summary["pooled"]["targets"]["tail_return"]["mae"]
        )


class TestResumeAndReportOnly:
    def test_report_only_aggregates_without_running_anything(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        run_sweep(tiny_settings, scenarios=(1,), seeds=(0,), parallel=1)
        cell = cell_settings(tiny_settings, 1, 0)
        stamp = diagnostics_paths(cell).summary.stat().st_mtime_ns
        again = run_sweep(
            tiny_settings, scenarios=(1,), seeds=(0,), parallel=1, execute=False
        )
        assert diagnostics_paths(cell).summary.stat().st_mtime_ns == stamp
        assert again["n_complete"] == 1

    def test_a_missing_cell_is_reported_not_invented(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        summary = run_sweep(
            tiny_settings, scenarios=(1, 2), seeds=(0,), parallel=1, execute=False
        )
        assert summary["n_cells"] == 2
        assert summary["n_complete"] == 0
        assert summary["missing"] == ["tiny_s1_seed0", "tiny_s2_seed0"]

    def test_resume_reuses_finished_cells(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        run_sweep(tiny_settings, scenarios=(1,), seeds=(0,), parallel=1)
        cell = cell_settings(tiny_settings, 1, 0)
        stamp = diagnostics_paths(cell).summary.stat().st_mtime_ns
        run_sweep(
            replace(tiny_settings, resume=True),
            scenarios=(1,),
            seeds=(0,),
            parallel=1,
        )
        assert diagnostics_paths(cell).summary.stat().st_mtime_ns == stamp
