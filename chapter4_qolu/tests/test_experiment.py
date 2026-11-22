"""Tests for the end-to-end joint experiment driver."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chapter4_qolu.experiment import (
    ExperimentSettings,
    main,
    resolve_raster,
    run_joint_scenario,
)


@pytest.fixture()
def settings(lcm_tif_path: Path, tmp_path: Path) -> ExperimentSettings:
    return ExperimentSettings(
        raster=lcm_tif_path,
        scenario=1,
        iterations=4,
        grid_size=16,
        stakeholders=2,
        seed=0,
        output_dir=tmp_path / "results",
    )


class TestRasterResolution:
    def test_accepts_an_explicit_path(self, lcm_tif_path: Path) -> None:
        assert resolve_raster(lcm_tif_path) == lcm_tif_path

    def test_falls_back_to_the_environment_variable(
        self, lcm_tif_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(os.environ, "CHAPTER4_LCM_TIF", str(lcm_tif_path))
        assert resolve_raster(None) == lcm_tif_path

    def test_unpacks_a_ukceh_download_zip(self, tmp_path: Path) -> None:
        import zipfile

        archive_path = tmp_path / "download.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("data/lcm2015_ni_1km.tif", b"not-a-real-raster")
        extracted = resolve_raster(archive_path)
        assert extracted.suffix == ".tif"
        assert extracted.is_file()

    def test_missing_raster_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_raster(tmp_path / "absent.tif")

    def test_no_raster_configured_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(os.environ, "CHAPTER4_LCM_TIF", raising=False)
        with pytest.raises(ValueError):
            resolve_raster(None)


class TestExperimentSettings:
    def test_defaults_follow_the_published_protocol(self, tmp_path: Path) -> None:
        settings = ExperimentSettings(raster=tmp_path / "raster.tif")
        assert settings.iterations == 1000
        assert settings.grid_size == 50
        assert settings.stakeholders == 3
        assert settings.scenario == 1
        assert settings.qolu_reward_mode == "per_parcel"

    def test_rejects_an_unknown_reward_mode(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            ExperimentSettings(
                raster=tmp_path / "raster.tif", qolu_reward_mode="global"
            )

    def test_rejects_an_unknown_scenario(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            ExperimentSettings(raster=tmp_path / "raster.tif", scenario=9)


class TestJointScenarioRun:
    def test_writes_a_json_record(self, settings: ExperimentSettings) -> None:
        record = run_joint_scenario(settings)
        output_path = Path(str(record["output_path"]))
        assert output_path.is_file()

        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert saved["scenario"] == 1
        assert saved["iterations"] == 4
        assert len(saved["per_iteration"]["team_reward"]) == 4
        assert len(saved["per_iteration"]["buia_reward"]) == 4
        assert set(saved["final_metrics"]) == {
            "land_use",
            "woodland",
            "urban",
            "conversion_alignment",
            "c_align",
            "well_being",
        }
        assert saved["raster"]["placeholder_region"] is True
        assert saved["runtime_seconds"] >= 0.0

    def test_is_reproducible(self, settings: ExperimentSettings) -> None:
        first = run_joint_scenario(settings)
        second = run_joint_scenario(settings)
        assert first["per_iteration"] == second["per_iteration"]


class TestScenarioWeightSwitches:
    """``--scale-rewards-by-weight`` and ``--weighted-consensus``."""

    def test_defaults_leave_both_switches_off(self, tmp_path: Path) -> None:
        settings = ExperimentSettings(raster=tmp_path / "raster.tif")
        assert settings.scale_by_scenario_weight is False
        assert settings.weighted_consensus is False

    def test_the_record_carries_the_switches(
        self, lcm_tif_path: Path, tmp_path: Path
    ) -> None:
        record = run_joint_scenario(
            ExperimentSettings(
                raster=lcm_tif_path,
                scenario=3,
                iterations=2,
                grid_size=12,
                stakeholders=1,
                output_dir=tmp_path / "results",
                scale_by_scenario_weight=True,
                weighted_consensus=True,
            )
        )
        assert record["scale_by_scenario_weight"] is True
        assert record["weighted_consensus"] is True

    def test_the_cli_flags_reach_the_written_record(
        self, lcm_tif_path: Path, tmp_path: Path
    ) -> None:
        assert (
            main(
                [
                    "--raster",
                    str(lcm_tif_path),
                    "--scenario",
                    "3",
                    "--iterations",
                    "2",
                    "--grid-size",
                    "12",
                    "--stakeholders",
                    "1",
                    "--scale-rewards-by-weight",
                    "--weighted-consensus",
                    "--output-dir",
                    str(tmp_path),
                ]
            )
            == 0
        )
        saved = json.loads(
            (tmp_path / "scenario3_joint_seed0.json").read_text(encoding="utf-8")
        )
        assert saved["scale_by_scenario_weight"] is True
        assert saved["weighted_consensus"] is True

    def test_the_cli_defaults_to_both_switches_off(
        self, lcm_tif_path: Path, tmp_path: Path
    ) -> None:
        main(
            [
                "--raster",
                str(lcm_tif_path),
                "--iterations",
                "2",
                "--grid-size",
                "12",
                "--stakeholders",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        saved = json.loads(
            (tmp_path / "scenario1_joint_seed0.json").read_text(encoding="utf-8")
        )
        assert saved["scale_by_scenario_weight"] is False
        assert saved["weighted_consensus"] is False
