"""Tests for the saturation / sensitivity sweep driver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chapter4_qolu.sweep_saturation import (
    RunOutcome,
    SweepSettings,
    first_saturated_iteration,
    render_markdown_table,
    run_sweep,
    sweep_combinations,
)


class TestSaturationDetection:
    def test_reports_the_first_iteration_above_the_threshold(self) -> None:
        assert first_saturated_iteration([0.1, 0.5, 0.995, 0.999], 0.99) == 2

    def test_reports_none_when_the_landscape_never_saturates(self) -> None:
        assert first_saturated_iteration([0.1, 0.2, 0.3], 0.99) is None

    def test_handles_an_empty_history(self) -> None:
        assert first_saturated_iteration([], 0.99) is None


class TestCombinations:
    def test_covers_the_full_grid_of_settings(self) -> None:
        combinations = sweep_combinations(
            grid_sizes=(50, 100),
            conversion_fractions=(1.0, 0.25),
            reward_modes=("team", "per_parcel"),
        )
        assert len(combinations) == 8
        assert len(set(combinations)) == 8
        assert (50, 1.0, "team") in combinations
        assert (100, 0.25, "per_parcel") in combinations


class TestMarkdownTable:
    def test_renders_one_row_per_run(self) -> None:
        outcomes = [
            RunOutcome(
                grid_size=50,
                conversion_fraction=1.0,
                reward_mode="team",
                saturation_iteration=434,
                approval_rate_early=0.52,
                approval_rate_late=0.62,
                applied_rate_early=0.5,
                applied_rate_late=0.18,
                total_conversions=2499,
                runtime_seconds=121.0,
                land_fraction=1.0,
                metrics={
                    "land_use": 0.08,
                    "woodland": 0.0,
                    "urban": 0.89,
                    "conversion_alignment": 0.049,
                    "well_being": 0.019,
                },
            ),
            RunOutcome(
                grid_size=100,
                conversion_fraction=0.25,
                reward_mode="per_parcel",
                saturation_iteration=None,
                approval_rate_early=0.3,
                approval_rate_late=0.1,
                applied_rate_early=0.25,
                applied_rate_late=0.09,
                total_conversions=900,
                runtime_seconds=200.0,
                land_fraction=0.5,
                metrics={
                    "land_use": 0.5,
                    "woodland": 0.9,
                    "urban": 0.5,
                    "conversion_alignment": 0.001,
                    "well_being": 0.5,
                },
            ),
        ]
        table = render_markdown_table(outcomes)
        lines = [line for line in table.splitlines() if line.startswith("|")]
        assert len(lines) == 2 + len(outcomes)  # header + separator + rows
        assert "never" in table
        assert "434" in table
        assert "per_parcel" in table
        assert "applied 200+" in table
        column_counts = {line.count("|") for line in lines}
        assert len(column_counts) == 1  # every row has the same column count

    def test_empty_sweep_still_renders_a_header(self) -> None:
        table = render_markdown_table([])
        assert table.strip().startswith("|")


class TestSweepRun:
    def test_writes_json_and_markdown(self, lcm_tif_path: Path, tmp_path: Path) -> None:
        settings = SweepSettings(
            raster=lcm_tif_path,
            iterations=3,
            seed=0,
            grid_sizes=(12,),
            conversion_fractions=(1.0,),
            reward_modes=("per_parcel", "team"),
            stakeholders=1,
            output_dir=tmp_path,
        )
        outcomes = run_sweep(settings)
        assert len(outcomes) == 2

        json_path = tmp_path / "saturation_sweep_seed0.json"
        markdown_path = tmp_path / "saturation_sweep_seed0.md"
        assert json_path.is_file()
        assert markdown_path.is_file()

        record = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(record["runs"]) == 2
        first = record["runs"][0]
        assert set(first["metrics"]) == {
            "land_use",
            "woodland",
            "urban",
            "conversion_alignment",
            "c_align",
            "well_being",
        }
        assert "approval_rate_early" in first
        assert "applied_rate_late" in first
        assert "total_conversions" in first
        assert set(first["phase_counts"]) == {"early", "late"}
        assert set(first["phase_counts"]["early"]) == {
            "candidates",
            "approved",
            "proposals",
            "conversions",
        }
        assert markdown_path.read_text(encoding="utf-8").count("\n|") >= 3

    def test_skips_grids_the_raster_cannot_supply(
        self, lcm_tif_path: Path, tmp_path: Path
    ) -> None:
        settings = SweepSettings(
            raster=lcm_tif_path,
            iterations=2,
            seed=0,
            grid_sizes=(12, 10_000),  # the second is far larger than any raster
            conversion_fractions=(1.0,),
            reward_modes=("per_parcel",),
            stakeholders=1,
            output_dir=tmp_path,
        )
        outcomes = run_sweep(settings)
        assert len(outcomes) == 1
        record = json.loads(
            (tmp_path / "saturation_sweep_seed0.json").read_text(encoding="utf-8")
        )
        assert record["skipped"]
        assert "10000" in json.dumps(record["skipped"])

    def test_rejects_an_empty_sweep(self, lcm_tif_path: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            SweepSettings(
                raster=lcm_tif_path,
                grid_sizes=(),
                output_dir=tmp_path,
            )
