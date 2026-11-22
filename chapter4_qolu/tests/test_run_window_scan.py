"""Tests for the window-scan driver and its report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.campaign_plan import CampaignSettings, campaign_runs
from chapter4_qolu.reporting.paths import parse_scale_config
from chapter4_qolu.reporting.scan_report import render_scan_report
from chapter4_qolu.run_window_scan import (
    PUBLISHED_CONFIG,
    ScanSettings,
    _build_parser,
    _settings_from,
    estimate_stage_seconds,
    metric_values_from_records,
    scan_paths,
    stage1_settings,
    stage2_settings,
)
from chapter4_qolu.scenario_worker import RunSpec, execute_run
from chapter4_qolu.tests.raster_fixtures import write_geotiff
from chapter4_qolu.window_scan import ScanWindow


@pytest.fixture()
def tiny_class_raster(tmp_path: Path) -> Path:
    """A 20 x 20 km class raster anchored at BNG (400 km E, 120 km N)."""
    rng = np.random.default_rng(20260924)
    codes = rng.integers(1, 11, size=(20, 20), dtype=np.uint8)
    return write_geotiff(
        tmp_path / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        codes,
        origin_e_m=400_000.0,
        origin_n_m=120_000.0,
    )


class TestRunSpecWindow:
    def test_rejects_two_window_origins_at_once(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not both"):
            RunSpec(
                raster=tmp_path / "raster.tif",
                method="mitm",
                scenario=1,
                seed=0,
                scale=parse_scale_config("10:1.0:10"),
                output_path=tmp_path / "out.json",
                window_origin=(0, 0),
                window_bng_km=(400.0, 120.0),
            )

    def test_record_carries_the_window(
        self, tiny_class_raster: Path, tmp_path: Path
    ) -> None:
        spec = RunSpec(
            raster=tiny_class_raster,
            method="mitm",
            scenario=1,
            seed=0,
            scale=parse_scale_config("8:1.0:10"),
            output_path=tmp_path / "mitm_seed0.json",
            iterations=2,
            stakeholders=1,
            window_bng_km=(405.0, 115.0),
            window_label="E405_N107",
        )
        record = execute_run(spec)
        assert record["window"]["label"] == "E405_N107"
        assert record["window"]["bng_top_left_km"] == [405.0, 115.0]
        assert record["window"]["raster_origin"] == [5, 5]

    def test_an_unpinned_run_records_the_window_it_actually_used(
        self, tiny_class_raster: Path, tmp_path: Path
    ) -> None:
        spec = RunSpec(
            raster=tiny_class_raster,
            method="mitm",
            scenario=1,
            seed=0,
            scale=parse_scale_config("8:1.0:10"),
            output_path=tmp_path / "mitm_seed0.json",
            iterations=2,
            stakeholders=1,
        )
        record = execute_run(spec)
        assert record["window"]["label"] is None
        assert len(record["window"]["raster_origin"]) == 2

    def test_pinned_windows_give_different_landscapes(
        self, tiny_class_raster: Path, tmp_path: Path
    ) -> None:
        digests = []
        for origin in ((0, 0), (10, 10)):
            spec = RunSpec(
                raster=tiny_class_raster,
                method="mitm",
                scenario=1,
                seed=0,
                scale=parse_scale_config("8:1.0:10"),
                output_path=tmp_path / f"mitm_{origin[0]}.json",
                iterations=2,
                stakeholders=1,
                window_origin=origin,
            )
            digests.append(execute_run(spec)["initial_grid_digest"])
        assert digests[0] != digests[1]


class TestCampaignWindowPlumbing:
    def test_campaign_runs_carry_the_window(self, tmp_path: Path) -> None:
        settings = CampaignSettings(
            raster=tmp_path / "raster.tif",
            tag="demo",
            scenarios=(1,),
            seeds=(0,),
            configs=(parse_scale_config("50:1.0:10"),),
            window_bng_km=(495.0, 190.0),
            window_label="E495_N140",
            output_dir=tmp_path,
        )
        specs = campaign_runs(settings)
        assert specs
        assert all(spec.window_bng_km == (495.0, 190.0) for spec in specs)
        assert all(spec.window_label == "E495_N140" for spec in specs)

    def test_metadata_reports_the_window(self, tmp_path: Path) -> None:
        settings = CampaignSettings(
            raster=tmp_path / "raster.tif",
            window_bng_km=(495.0, 190.0),
            window_label="E495_N140",
            output_dir=tmp_path,
        )
        metadata = settings.metadata()
        assert metadata["window_label"] == "E495_N140"
        assert metadata["window_bng_top_left_km"] == [495.0, 190.0]


class TestStageSettings:
    @pytest.fixture()
    def settings(self, tmp_path: Path) -> ScanSettings:
        return ScanSettings(raster=tmp_path / "raster.img", output_dir=tmp_path)

    def test_stage1_is_scenario_one_seed_zero_unweighted(
        self, settings: ScanSettings
    ) -> None:
        window = ScanWindow(easting_km=495, northing_km=140)
        campaign = stage1_settings(settings, window)
        assert campaign.scenarios == (1,)
        assert campaign.seeds == (0,)
        assert campaign.methods == ("qolu_buia", "pmaddpg", "ltos", "mitm")
        assert [scale.label for scale in campaign.configs] == ["50x50_f1.0_cap10"]
        assert campaign.scale_by_scenario_weight is False
        assert campaign.weighted_consensus is False
        assert campaign.qolu_reward_mode == "per_parcel"
        assert campaign.window_bng_km == (495.0, 190.0)
        assert campaign.campaign_dir.parent == scan_paths(settings).stage1

    def test_stage2_is_all_scenarios_three_seeds_weighted(
        self, settings: ScanSettings
    ) -> None:
        window = ScanWindow(easting_km=495, northing_km=140)
        campaign = stage2_settings(settings, window)
        assert campaign.scenarios == (1, 2, 3, 4)
        assert campaign.seeds == (0, 1, 2)
        assert campaign.scale_by_scenario_weight is True
        assert campaign.weighted_consensus is True
        assert campaign.campaign_dir.parent == scan_paths(settings).stage2

    def test_stage2_seed_count_is_configurable(self, tmp_path: Path) -> None:
        settings = ScanSettings(
            raster=tmp_path / "raster.img", output_dir=tmp_path, stage2_seeds=(0, 1)
        )
        campaign = stage2_settings(settings, ScanWindow(easting_km=420, northing_km=90))
        assert campaign.seeds == (0, 1)

    def test_estimates_are_positive_and_scale_with_the_work(
        self, settings: ScanSettings
    ) -> None:
        windows = (
            ScanWindow(easting_km=495, northing_km=140),
            ScanWindow(easting_km=520, northing_km=140),
        )
        one = estimate_stage_seconds(settings, windows[:1], stage=1)
        two = estimate_stage_seconds(settings, windows, stage=1)
        assert one > 0.0
        assert two == pytest.approx(2.0 * one)
        assert estimate_stage_seconds(settings, windows[:1], stage=2) > one


class TestConfigFlag:
    """``--config`` overrides :data:`PUBLISHED_CONFIG` for both stages."""

    def _settings(self, tmp_path: Path, *flags: str) -> ScanSettings:
        arguments = _build_parser().parse_args(
            ["--report-only", "--output-dir", str(tmp_path), *flags]
        )
        return _settings_from(arguments)

    def test_defaults_to_the_published_config(self, tmp_path: Path) -> None:
        assert self._settings(tmp_path).config == PUBLISHED_CONFIG

    def test_overrides_the_config(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path, "--config", "50:1.0:1")
        assert settings.config.max_conversions == 1

    def test_stage1_campaign_carries_the_chosen_config(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path, "--config", "50:1.0:1")
        window = ScanWindow(easting_km=495, northing_km=140)
        campaign = stage1_settings(settings, window)
        assert [scale.label for scale in campaign.configs] == ["50x50_f1.0_cap1"]


class TestMetricValuesFromRecords:
    def test_pivots_records_into_metric_by_method(self) -> None:
        records = [
            {"method": "mitm", "final_metrics": {"land_use": 0.2, "woodland": 0.1}},
            {"method": "qolu_buia", "final_metrics": {"land_use": 0.4}},
        ]
        values = metric_values_from_records(records)
        assert values["land_use"] == {"mitm": 0.2, "qolu_buia": 0.4}
        assert values["woodland"] == {"mitm": 0.1}

    def test_ignores_metrics_outside_the_published_three(self) -> None:
        records = [{"method": "mitm", "final_metrics": {"well_being": 0.5}}]
        assert metric_values_from_records(records) == {}


def _scan_summary() -> dict:
    window = ScanWindow(easting_km=495, northing_km=140)
    other = ScanWindow(easting_km=520, northing_km=140)
    return {
        "generated": "2026-09-24T12:00:00+00:00",
        "raster": {"name": "lcm2015_gb_1km_dominant_aggregate_class.img"},
        "grid": {
            "bounding_box_km": {
                "east_min": 420,
                "east_max": 640,
                "north_min": 90,
                "north_max": 250,
            },
            "window_size_km": 50,
            "stride_km": 25,
            "max_sea_fraction": 0.3,
            "n_candidates": 40,
            "n_retained": 2,
            "eastings": [495, 520],
            "northings": [140],
        },
        "dropped": [],
        "stage1": {
            "iterations": 1000,
            "score_distribution": {"0": 1, "1": 0, "2": 0, "3": 1},
            "windows": [
                {
                    "window": window.as_record(),
                    "composition": {
                        "n_cells": 2500,
                        "sea_fraction": 0.0,
                        "class_shares": {"4": 0.5, "10": 0.5},
                    },
                    "score": {
                        "first_place_count": 3,
                        "kendall_sum": 3.0,
                        "reference_minmax_mean": 1.0,
                        "metrics": [],
                    },
                    "methods": {
                        "qolu_buia": {
                            "metrics": {"land_use": 0.05},
                            "total_conversions": 2500,
                            "saturation_iteration": 400,
                        }
                    },
                },
                {
                    "window": other.as_record(),
                    "composition": {
                        "n_cells": 2500,
                        "sea_fraction": 0.1,
                        "class_shares": {"3": 1.0},
                    },
                    "score": {
                        "first_place_count": 0,
                        "kendall_sum": -3.0,
                        "reference_minmax_mean": 0.0,
                        "metrics": [],
                    },
                    "methods": {},
                },
            ],
        },
        "stage2": {
            "seeds": [0, 1, 2],
            "windows": [
                {
                    "window": window.as_record(),
                    "matched_cells": 7,
                    "total_cells": 12,
                    "cells": [],
                    "summary": {"configurations": []},
                }
            ],
            "winner": window.as_record(),
        },
        "selection": {
            "n_candidate_windows": 40,
            "n_scanned_windows": 2,
            "winner": window.as_record(),
        },
    }


class TestScanReport:
    def test_states_the_bounding_box_and_the_grid(self) -> None:
        text = render_scan_report(_scan_summary())
        assert "420" in text and "640" in text
        assert "50 x 50 km" in text or "50 × 50 km" in text

    def test_lists_every_retained_window(self) -> None:
        text = render_scan_report(_scan_summary())
        assert "E495_N140" in text
        assert "E520_N140" in text

    def test_reports_the_score_distribution(self) -> None:
        text = render_scan_report(_scan_summary())
        assert "Score distribution" in text

    def test_marks_a_flat_metric_instead_of_an_alphabetical_ranking(self) -> None:
        summary = _scan_summary()
        summary["stage1"]["windows"][0]["score"]["metrics"] = [
            {
                "metric": "woodland",
                "ranking": ["ltos", "mitm", "pmaddpg", "qolu_buia"],
                "reference_first": False,
                "kendall_tau": 0.0,
                "reference_minmax": 0.5,
                "flat": True,
            }
        ]
        summary["stage1"]["windows"][0]["score"]["flat_metrics"] = 1
        text = render_scan_report(summary)
        assert "flat: all four identical" in text
        assert "LToS > MitM > P-MADDPG > QOLU+BUIA" not in text

    def test_prints_tied_methods_with_an_equals_sign(self) -> None:
        summary = _scan_summary()
        summary["stage1"]["windows"][0]["score"]["metrics"] = [
            {
                "metric": "land_use",
                "ranking": ["pmaddpg", "ltos", "mitm", "qolu_buia"],
                "groups": [["pmaddpg"], ["ltos", "mitm", "qolu_buia"]],
                "reference_first": False,
                "reference_joint_first": False,
                "kendall_tau": 0.167,
                "reference_minmax": 0.0,
                "flat": False,
                "tied_pairs": 3,
            }
        ]
        text = render_scan_report(summary)
        assert "P-MADDPG > LToS = MitM = QOLU+BUIA" in text

    def test_reports_how_many_windows_separated_no_methods(self) -> None:
        summary = _scan_summary()
        summary["stage1"]["windows"][0]["score"]["flat_metrics"] = 3
        summary["stage1"]["windows"][1]["score"]["flat_metrics"] = 0
        text = render_scan_report(summary)
        assert "1 of the 2 scanned windows" in text
        assert "identical" in text

    def test_flat_windows_explain_the_cap_from_the_configuration(self) -> None:
        """cap10: at least one flat window -> the saturation explanation,
        with the cap read from ``stage1.configuration`` rather than
        hard-coded."""
        summary = _scan_summary()
        summary["stage1"]["configuration"] = "50x50_f1.0_cap10"
        summary["stage1"]["windows"][0]["score"]["flat_metrics"] = 3
        text = render_scan_report(summary)
        assert "1 of the 2 scanned windows all four" in text
        assert "up to 10 conversions per iteration" in text
        assert "long before iteration 1000" in text

    def test_no_flat_windows_point_at_the_conversions_table_instead(self) -> None:
        """cap1: no window is fully flat, so the paragraph must not claim a
        saturation cause it did not observe, and must not hard-code cap10's
        explanation."""
        summary = _scan_summary()
        summary["stage1"]["configuration"] = "50x50_f1.0_cap1"
        text = render_scan_report(summary)
        assert "0 of the 2 scanned windows all four" in text
        assert "methods differ on every scanned window" in text
        assert "configuration `50x50_f1.0_cap1`" in text
        assert "up to" not in text
        assert "long before iteration 1000" not in text

    def test_saturation_table_states_windows_below_half_the_best(self) -> None:
        """Surfaces P-MADDPG's bimodal conversion behaviour: below-50%
        windows are counted from the same per-method records that build the
        saturation table, not asserted separately."""
        summary = _scan_summary()
        summary["stage1"]["windows"][0]["methods"] = {
            "ltos": {"total_conversions": 1000, "saturation_iteration": None},
            "pmaddpg": {"total_conversions": 180, "saturation_iteration": None},
            "mitm": {"total_conversions": 1000, "saturation_iteration": None},
            "qolu_buia": {"total_conversions": 920, "saturation_iteration": 400},
        }
        summary["stage1"]["windows"][1]["methods"] = {
            "ltos": {"total_conversions": 1000, "saturation_iteration": None},
            "pmaddpg": {"total_conversions": 900, "saturation_iteration": None},
            "mitm": {"total_conversions": 1000, "saturation_iteration": None},
            "qolu_buia": {"total_conversions": 950, "saturation_iteration": 500},
        }
        text = render_scan_report(summary)
        assert (
            "Total conversions were below 50% of the best-converting "
            "method's on 0 (LToS), 1 (P-MADDPG), 0 (MitM) and "
            "0 (QOLU+BUIA) of the 2 windows."
        ) in text

    def test_the_method_section_states_the_primary_score_reached(self) -> None:
        text = render_scan_report(_scan_summary())
        # One window reached 3/3 in the fixture, so the report has to say so
        # rather than leave the reader to infer it from the table.
        assert "best primary score" in text

    def test_contains_the_method_section(self) -> None:
        text = render_scan_report(_scan_summary())
        assert "Method - how the window was chosen" in text
        assert "every window is listed" in text
        assert "40" in text
        assert "the *ordering* only" in text

    def test_shortens_a_wkt_crs_to_its_name(self) -> None:
        summary = _scan_summary()
        summary["raster"]["crs"] = (
            'PROJCS["Transverse Mercator",GEOGCS["Ord. Survey G. Britain 1936",'
            'DATUM["Ord. Survey G. Britain 1936",SPHEROID["Airy",6377563.396,'
            "299.324975315035]]]]"
        )
        text = render_scan_report(summary)
        assert "Transverse Mercator" in text
        assert "SPHEROID" not in text

    def test_keeps_a_short_crs_as_it_is(self) -> None:
        summary = _scan_summary()
        summary["raster"]["crs"] = "EPSG:27700"
        assert "EPSG:27700" in render_scan_report(summary)

    def test_renders_without_a_stage_two(self) -> None:
        summary = _scan_summary()
        summary["stage2"] = {"seeds": [], "windows": [], "winner": None}
        text = render_scan_report(summary)
        assert "Method - how the window was chosen" in text

    def test_stage2_overview_states_the_configured_scale(self) -> None:
        summary = _scan_summary()
        summary["stage1"]["configuration"] = "50x50_f1.0_cap1"
        summary["stage2"]["configuration"] = "50x50_f1.0_cap1"
        text = render_scan_report(summary)
        assert "configuration `50x50_f1.0_cap1`" in text
        assert "50x50_f1.0_cap10" not in text

    def test_round_trips_through_json(self) -> None:
        summary = _scan_summary()
        assert json.loads(json.dumps(summary)) == summary
