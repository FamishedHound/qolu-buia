"""End-to-end tests for the sensitivity sweep and its report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.reporting.sensitivity_report import render_sensitivity_report
from chapter4_qolu.run_sensitivity import (
    SensitivitySettings,
    _build_parser,
    _settings_from,
    run_sensitivity,
)
from chapter4_qolu.sensitivity_plan import sensitivity_paths
from chapter4_qolu.tests.raster_fixtures import write_geotiff


@pytest.fixture()
def tiny_class_raster(tmp_path: Path) -> Path:
    rng = np.random.default_rng(20260924)
    codes = rng.integers(1, 11, size=(24, 24), dtype=np.uint8)
    return write_geotiff(
        tmp_path / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        codes,
        origin_e_m=400_000.0,
        origin_n_m=124_000.0,
    )


@pytest.fixture()
def tiny_settings(tiny_class_raster: Path, tmp_path: Path) -> SensitivitySettings:
    """A sweep small enough to run inside a test: two tiny environments."""
    return SensitivitySettings(
        raster=tiny_class_raster,
        tag="tiny",
        environments=("E402_N110_6x6", "E404_N112_6x8"),
        caps=(1, 2),
        reward_modes=("per_parcel",),
        pmaddpg_variants=("pmaddpg",),
        reference_runs=(),
        iterations=3,
        stakeholders=1,
        stage_b_seeds=(0, 1),
        stage_b_scenarios=(1, 2),
        stage_b_configurations=1,
        stage_c_seeds=(0, 1),
        parallel=1,
        output_dir=tmp_path / "results",
    )


class TestDryRun:
    def test_a_dry_run_writes_nothing(self, tiny_settings: SensitivitySettings) -> None:
        run_sensitivity(tiny_settings, dry_run=True)
        paths = sensitivity_paths(tiny_settings)
        assert not paths.summary.exists()
        assert not paths.report.exists()


class TestEndToEnd:
    def test_the_sweep_produces_a_summary_and_a_report(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        summary = run_sensitivity(tiny_settings)
        paths = sensitivity_paths(tiny_settings)
        assert paths.summary.is_file()
        assert paths.report.is_file()
        assert summary["stage_a"]["n_configurations"] == 2 * 2 * 1 * 1 * 2 * 2
        assert summary["stage_a"]["configurations"]

    def test_every_stage_a_configuration_is_reported(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        summary = run_sensitivity(tiny_settings)
        reported = summary["stage_a"]["configurations"]
        assert len(reported) == summary["stage_a"]["n_configurations"]

    def test_stage_b_writes_the_five_campaign_documents(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        run_sensitivity(tiny_settings)
        paths = sensitivity_paths(tiny_settings)
        campaigns = sorted(paths.stage_b.glob("campaign_*"))
        assert campaigns
        for campaign in campaigns:
            for name in (
                "summary.json",
                "tables.md",
                "tables.tex",
                "pattern_check.md",
                "per_seed.csv",
            ):
                assert (campaign / name).is_file()

    def test_both_consensus_readings_are_run(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        run_sensitivity(tiny_settings)
        paths = sensitivity_paths(tiny_settings)
        tags = {path.name for path in paths.stage_b.glob("campaign_*")}
        assert any(tag.endswith("_plain") for tag in tags)
        assert any(tag.endswith("_weighted") for tag in tags)

    def test_stage_c_writes_the_deliverable_campaign(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        summary = run_sensitivity(tiny_settings)
        paths = sensitivity_paths(tiny_settings)
        assert summary["stage_c"] is not None
        assert summary["stage_c"]["seeds"] == [0, 1]
        campaigns = sorted(paths.stage_c.glob("campaign_*"))
        assert len(campaigns) == 1
        assert (campaigns[0] / "per_seed.csv").is_file()

    def test_stage_b_reports_both_match_counts(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        summary = run_sensitivity(tiny_settings)
        for entry in summary["stage_b"]["campaigns"]:
            assert "matched_cells_tie_aware" in entry
            assert "matched_cells_alphabetical" in entry
            assert entry["total_cells"] > 0

    def test_a_resumed_sweep_reruns_nothing(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        from dataclasses import replace

        run_sensitivity(tiny_settings)
        paths = sensitivity_paths(tiny_settings)
        stamps = {
            path: path.stat().st_mtime_ns
            for path in paths.stage_a.rglob("*_seed*.json")
        }
        assert stamps
        run_sensitivity(replace(tiny_settings, resume=True))
        for path, stamp in stamps.items():
            assert path.stat().st_mtime_ns == stamp

    def test_the_report_states_how_the_configuration_was_chosen(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        summary = run_sensitivity(tiny_settings)
        report = render_sensitivity_report(summary)
        assert "Method - how the configuration was chosen" in report
        assert "every configuration is listed" in report
        assert "the ordering" in report

    def test_the_report_lists_every_configuration(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        summary = run_sensitivity(tiny_settings)
        report = render_sensitivity_report(summary)
        for entry in summary["stage_a"]["configurations"]:
            assert entry["configuration"]["label"] in report

    def test_the_summary_records_the_normalisation_finding(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        summary = run_sensitivity(tiny_settings)
        assert summary["published"]["minmax_compatible"] is False
        report = render_sensitivity_report(summary)
        assert "min-max" in report.lower()


class TestCli:
    def test_the_parser_accepts_pinned_environments(self, tmp_path: Path) -> None:
        arguments = _build_parser().parse_args(
            [
                "--raster",
                "x.img",
                "--environments",
                "SE_region",
                "E495_N140",
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert arguments.environments == ["SE_region", "E495_N140"]

    def test_the_settings_carry_the_pinned_environments(
        self, tiny_class_raster: Path, tmp_path: Path
    ) -> None:
        arguments = _build_parser().parse_args(
            [
                "--raster",
                str(tiny_class_raster),
                "--windows",
                "E495_N140",
                "--tag",
                "cli",
                "--output-dir",
                str(tmp_path),
            ]
        )
        settings = _settings_from(arguments)
        assert settings.environments == ("E495_N140",)
        assert settings.tag == "cli"

    def test_the_default_environments_are_the_documented_three(
        self, tiny_class_raster: Path, tmp_path: Path
    ) -> None:
        arguments = _build_parser().parse_args(
            ["--raster", str(tiny_class_raster), "--output-dir", str(tmp_path)]
        )
        settings = _settings_from(arguments)
        assert settings.environments == ("SE_region", "E495_N140", "E495_N165")


class TestSummaryShape:
    def test_the_summary_is_json_serialisable(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        summary = run_sensitivity(tiny_settings)
        assert json.loads(json.dumps(summary))["tag"] == "tiny"


class TestSelectedConfiguration:
    """The report must name the configuration the tables rank first.

    The stage-A fallback has to be the *ranked* first configuration:
    ``scores[0]`` is the first cell in *configuration order* and means nothing,
    and would name a configuration that the "best configurations" table does
    not even list.
    """

    @staticmethod
    def _first_row_label(report: str) -> str:
        """The configuration label in the first data row of the top table."""
        lines = report.splitlines()
        start = lines.index("### The best configurations, by coordinate")
        row = next(line for line in lines[start + 4 :] if line.startswith("| 1 |"))
        # | 1 | `env` | cap | `mode` | `pmaddpg` | `ltos` | `mitm` | ...
        cells = [cell.strip().strip("`") for cell in row.split("|")]
        environment, cap, mode, pmaddpg, ltos, mitm = cells[2:8]
        return f"{environment}_cap{cap}_{mode}_{pmaddpg}_{ltos}_{mitm}"

    def test_stage_a_only_names_the_top_ranked_configuration(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        summary = run_sensitivity(tiny_settings, stages=("a",))
        assert summary["selection"]["winner_stage"] == "A"
        assert not summary["stage_b"]["campaigns"]
        report = render_sensitivity_report(summary)
        assert summary["selection"]["winner"]["label"] == self._first_row_label(report)
        assert "stage B has not been run" in report

    def test_after_stage_b_the_winner_is_named_with_its_stage_a_row(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        summary = run_sensitivity(tiny_settings)
        assert summary["selection"]["winner_stage"] == "B"
        winner = summary["selection"]["winner"]["label"]
        assert winner == summary["stage_b"]["campaigns"][0]["configuration"]["label"]
        report = render_sensitivity_report(summary)
        assert f"`{winner}`" in report
        assert "Its stage-A screening row" in report
        # With one stage-B configuration the finalist is also the stage-A #1.
        assert winner == self._first_row_label(report)

    def test_the_stage_b_table_carries_the_stage_a_rank_and_score(
        self, tiny_settings: SensitivitySettings
    ) -> None:
        summary = run_sensitivity(tiny_settings)
        report = render_sensitivity_report(summary)
        assert "| Stage-A rank | Configuration | Stage-A score |" in report

    def test_the_whole_region_paragraph_appears_when_it_was_swept(
        self, tiny_class_raster: Path, tmp_path: Path
    ) -> None:
        from dataclasses import replace

        settings = replace(
            SensitivitySettings(
                raster=tiny_class_raster,
                tag="region_note",
                environments=("E402_N110_6x6",),
                caps=(1,),
                reward_modes=("per_parcel",),
                pmaddpg_variants=("pmaddpg",),
                reference_runs=(),
                iterations=2,
                stakeholders=1,
                stage_b_configurations=1,
                stage_b_seeds=(0,),
                stage_b_scenarios=(1,),
                stage_c_seeds=(0,),
                parallel=1,
                output_dir=tmp_path / "results",
            )
        )
        summary = run_sensitivity(settings, stages=("a",))
        report = render_sensitivity_report(summary)
        # No SE_region in this sweep, so the paragraph must be absent rather
        # than present and empty.
        assert "The whole-region rows" not in report


class TestPartialStageB:
    """An unfinished stage-B campaign must not read as a finished failure."""

    @staticmethod
    def _summary(n_runs: int) -> dict:
        return {
            "tag": "partial",
            "settings": {},
            "published": {},
            "stage_a": {
                "axes": {},
                "windows": [],
                "configurations": [],
                "score_distribution": {},
                "n_configurations": 1,
                "n_cells": 1,
            },
            "reference": [],
            "stage_b": {
                "seeds": [0, 1, 2],
                "scenarios": [1, 2, 3, 4],
                "choices": [],
                "campaigns": [
                    {
                        "tag": "b1_demo_plain",
                        "rank": 1,
                        "configuration": {"label": "demo", "description": ""},
                        "window": {"label": "W", "description": ""},
                        "reading": "plain",
                        "weighted": False,
                        "matched_cells_tie_aware": 0,
                        "matched_cells_alphabetical": 0,
                        "total_cells": 12,
                        "value_mae": {"absolute": 0.3},
                        "closeness_view": "absolute",
                        "n_runs": n_runs,
                        "summary": {"configurations": []},
                        "campaign_dir": "x",
                    }
                ],
            },
            "stage_c": None,
            "selection": {"winner": {"label": "demo"}, "winner_stage": "B"},
        }

    def test_a_short_campaign_is_flagged(self) -> None:
        report = render_sensitivity_report(self._summary(12))
        assert "**12 of 48**" in report
        assert "Some campaigns above are incomplete" in report

    def test_a_complete_campaign_is_not_flagged(self) -> None:
        report = render_sensitivity_report(self._summary(48))
        assert "48 of 48" in report
        assert "**48 of 48**" not in report
        assert "Some campaigns above are incomplete" not in report
