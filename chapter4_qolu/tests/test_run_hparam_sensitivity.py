"""End-to-end tests for the hyper-parameter sensitivity analysis and its report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chapter4_qolu.hparam_plan import (
    BASELINE_TAG,
    HParamSettings,
    effective_perturbations,
    hparam_paths,
    settings_perturbations,
)
from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS
from chapter4_qolu.reporting.hparam_report import (
    HPARAM_RUN_COLUMNS,
    render_hparam_report,
    render_hparam_runs_csv,
)
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.run_hparam_sensitivity import (
    _build_parser,
    _settings_from,
    main,
    run_hparam_sensitivity,
)
from chapter4_qolu.tests.raster_fixtures import write_geotiff


@pytest.fixture(scope="module")
def tiny_class_raster(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(20260929)
    codes = rng.integers(1, 11, size=(24, 24), dtype=np.uint8)
    return write_geotiff(
        tmp_path_factory.mktemp("hparam_raster")
        / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        codes,
        origin_e_m=400_000.0,
        origin_n_m=124_000.0,
    )


@pytest.fixture(scope="module")
def tiny_settings(
    tiny_class_raster: Path, tmp_path_factory: pytest.TempPathFactory
) -> HParamSettings:
    """An analysis small enough to run inside a test: two methods, two knobs.

    Module-scoped, and so is :func:`summary`: the analysis is executed once and
    every assertion reads the same records, which is also what ``--resume``
    promises a real invocation.
    """
    return HParamSettings(
        raster=tiny_class_raster,
        tag="tiny",
        hyperparameters=("gamma", "samples_per_iteration"),
        methods=("qolu_buia", "mitm"),
        scenarios=(1,),
        seeds=(0, 1, 2),
        iterations=3,
        stakeholders=1,
        scale=ScaleConfig(grid_size=6, conversion_fraction=1.0, max_conversions=2),
        window="E402_N110_6x6",
        parallel=1,
        output_dir=tmp_path_factory.mktemp("hparam_results"),
    )


@pytest.fixture(scope="module")
def summary(tiny_settings: HParamSettings) -> dict[str, Any]:
    return run_hparam_sensitivity(tiny_settings)


class TestDryRun:
    def test_a_dry_run_writes_nothing(self, tiny_settings: HParamSettings) -> None:
        run_hparam_sensitivity(tiny_settings, dry_run=True)
        paths = hparam_paths(tiny_settings)
        assert not paths.summary.exists()
        assert not paths.report.exists()
        assert not paths.runs_csv.exists()


class TestEndToEnd:
    def test_the_three_documents_are_written(
        self, tiny_settings: HParamSettings, summary: dict[str, Any]
    ) -> None:
        paths = hparam_paths(tiny_settings)
        assert paths.summary.is_file()
        assert paths.report.is_file()
        assert paths.runs_csv.is_file()
        assert json.loads(paths.summary.read_text(encoding="utf-8"))["tag"] == "tiny"

    def test_the_baseline_campaign_holds_every_method(
        self, tiny_settings: HParamSettings, summary: dict[str, Any]
    ) -> None:
        paths = hparam_paths(tiny_settings)
        directory = paths.runs / f"campaign_{BASELINE_TAG}"
        assert directory.is_dir()
        assert len(list(directory.glob("*/scenario*/*.json"))) == 2 * 3

    def test_every_listed_perturbation_is_reported(
        self, tiny_settings: HParamSettings, summary: dict[str, Any]
    ) -> None:
        listed = settings_perturbations(tiny_settings)
        assert len(summary["perturbations"]) == len(listed) == 4
        labels = [item["label"] for item in summary["perturbations"]]
        assert labels == [item.label for item in listed]

    def test_an_agent_side_perturbation_only_reran_the_learner(
        self, tiny_settings: HParamSettings, summary: dict[str, Any]
    ) -> None:
        paths = hparam_paths(tiny_settings)
        directory = paths.runs / "campaign_gamma_minus"
        methods = {path.name.split("_seed")[0] for path in directory.glob("*/*/*.json")}
        assert methods == {"qolu_buia"}

    def test_the_comparison_still_has_all_four_rows(
        self, summary: dict[str, Any]
    ) -> None:
        block = next(
            item for item in summary["perturbations"] if item["label"] == "gamma_minus"
        )
        cell = block["scenarios"][0]["metrics"]["land_use"]
        assert set(cell["methods"]) == {"qolu_buia", "mitm"}
        assert cell["methods"]["mitm"]["borrowed_from_baseline"] is True
        assert cell["methods"]["qolu_buia"]["borrowed_from_baseline"] is False

    def test_each_method_reports_a_mean_and_a_standard_error(
        self, summary: dict[str, Any]
    ) -> None:
        for block in summary["perturbations"]:
            for scenario in block["scenarios"]:
                for metric in PUBLISHED_METRICS:
                    for entry in scenario["metrics"][metric]["methods"].values():
                        assert entry["n"] == 3
                        assert isinstance(entry["mean"], float)
                        assert isinstance(entry["se"], float)
                        assert "paired" in entry

    def test_every_cell_carries_a_ranking_verdict_and_a_tau(
        self, summary: dict[str, Any]
    ) -> None:
        for block in summary["perturbations"]:
            for scenario in block["scenarios"]:
                for metric in PUBLISHED_METRICS:
                    cell = scenario["metrics"][metric]
                    assert len(cell["ranking"]) == 2
                    assert len(cell["baseline_ranking"]) == 2
                    assert isinstance(cell["ranking_preserved"], bool)
                    assert -1.0 <= cell["kendall_tau"] <= 1.0
                    assert "absolute" in cell

    def test_the_verdict_agrees_with_the_two_rankings(
        self, summary: dict[str, Any]
    ) -> None:
        for block in summary["perturbations"]:
            for scenario in block["scenarios"]:
                for metric in PUBLISHED_METRICS:
                    cell = scenario["metrics"][metric]
                    same = cell["ranking"] == cell["baseline_ranking"]
                    assert cell["ranking_preserved"] is same
                    if same:
                        assert cell["kendall_tau"] == pytest.approx(1.0)

    def test_the_overall_table_counts_every_effective_perturbation(
        self, tiny_settings: HParamSettings, summary: dict[str, Any]
    ) -> None:
        overall = summary["overall"]
        assert overall["n_perturbations"] == 4
        assert overall["n_effective"] == len(effective_perturbations(tiny_settings))
        assert overall["n_no_op"] == 0
        for metric in PUBLISHED_METRICS:
            entry = overall["metrics"][metric]
            assert entry["total"] == overall["n_effective"]
            assert 0 <= entry["preserved"] <= entry["total"]
            assert -1.0 <= entry["mean_kendall_tau"] <= 1.0

    def test_the_run_csv_has_one_row_per_record(
        self, tiny_settings: HParamSettings, summary: dict[str, Any]
    ) -> None:
        paths = hparam_paths(tiny_settings)
        lines = paths.runs_csv.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == ",".join(HPARAM_RUN_COLUMNS)
        # baseline 2x3, gamma +/- 1x3 each, samples +/- 2x3 each
        assert len(lines) - 1 == 6 + 3 + 3 + 6 + 6
        assert any(row.startswith("gamma_minus,") for row in lines[1:])
        assert any(row.startswith(f"{BASELINE_TAG},") for row in lines[1:])

    def test_the_perturbed_value_reached_the_run_record(
        self, tiny_settings: HParamSettings, summary: dict[str, Any]
    ) -> None:
        paths = hparam_paths(tiny_settings)
        path = next(
            (paths.runs / "campaign_samples_per_iteration_plus").glob("*/*/*.json")
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["hyperparameter"]["hyperparameter"] == "samples_per_iteration"
        assert record["env_settings"]["samples_per_iteration"] == 60

    def test_an_agent_side_record_names_its_override(
        self, tiny_settings: HParamSettings, summary: dict[str, Any]
    ) -> None:
        paths = hparam_paths(tiny_settings)
        path = next((paths.runs / "campaign_gamma_plus").glob("*/*/*.json"))
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["qolu_overrides"] == {"gamma": 1.0}
        assert record["hyperparameter"]["clamped"] is True

    def test_the_report_names_the_promise_it_is_keeping(
        self, tiny_settings: HParamSettings, summary: dict[str, Any]
    ) -> None:
        text = hparam_paths(tiny_settings).report.read_text(encoding="utf-8")
        assert "+/-20" in text or "20 %" in text
        assert "Kendall" in text
        assert "gamma" in text
        assert "ranking" in text.lower()

    def test_report_only_rebuilds_without_a_raster(
        self, tiny_settings: HParamSettings, summary: dict[str, Any]
    ) -> None:
        paths = hparam_paths(tiny_settings)
        paths.report.unlink()
        again = run_hparam_sensitivity(tiny_settings, execute=False)
        assert paths.report.is_file()
        assert again["overall"] == summary["overall"]

    def test_resuming_runs_nothing_a_second_time(
        self, tiny_settings: HParamSettings, summary: dict[str, Any]
    ) -> None:
        from dataclasses import replace

        paths = hparam_paths(tiny_settings)
        before = {
            path: path.stat().st_mtime_ns
            for path in sorted(paths.runs.glob("*/*/*/*.json"))
        }
        run_hparam_sensitivity(replace(tiny_settings, resume=True))
        after = {
            path: path.stat().st_mtime_ns
            for path in sorted(paths.runs.glob("*/*/*/*.json"))
        }
        assert before == after


class TestThePerturbationReachesTheLearner:
    """The one thing the whole driver rests on: the agent is really retuned."""

    def _captured_config(
        self,
        settings: HParamSettings,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        direction: int,
    ) -> Any:
        from chapter4_qolu import hparam_worker
        from chapter4_qolu.agent.qolu_agent import QOLUAgent
        from chapter4_qolu.hparam_plan import perturbation_runs
        from chapter4_qolu.hparam_sensitivity import perturb, resolve_hyperparameters

        seen: list[Any] = []

        def _spy(config: Any, seed: int = 0) -> QOLUAgent:
            seen.append(config)
            return QOLUAgent(config, seed=seed)

        monkeypatch.setattr(hparam_worker, "QOLUAgent", _spy)
        item = perturb(resolve_hyperparameters([name])[0], direction)
        spec = perturbation_runs(settings, item)[0]
        hparam_worker.execute_hparam_run(spec)
        assert seen
        return seen[0]

    def test_a_perturbed_gamma_reaches_the_qolu_config(
        self,
        tiny_settings: HParamSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = self._captured_config(tiny_settings, monkeypatch, "gamma", -1)
        assert config.gamma == pytest.approx(0.99 * 0.8)

    def test_an_unperturbed_run_keeps_the_package_default(
        self,
        tiny_settings: HParamSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chapter4_qolu import hparam_worker
        from chapter4_qolu.agent.qolu_agent import QOLUAgent
        from chapter4_qolu.config import QOLUConfig
        from chapter4_qolu.hparam_plan import baseline_runs

        seen: list[Any] = []

        def _spy(config: Any, seed: int = 0) -> QOLUAgent:
            seen.append(config)
            return QOLUAgent(config, seed=seed)

        monkeypatch.setattr(hparam_worker, "QOLUAgent", _spy)
        spec = next(
            item for item in baseline_runs(tiny_settings) if item.method == "qolu_buia"
        )
        hparam_worker.execute_hparam_run(spec)
        # The baseline goes through scenario_worker.execute_run, which the spy
        # does not sit on, so nothing is captured and the default is untouched.
        assert seen == []
        assert QOLUConfig(observation_channels=1, observation_window=9).gamma == 0.99

    def test_an_environment_perturbation_reaches_the_observation_window(
        self,
        tiny_settings: HParamSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chapter4_qolu.hparam_plan import perturbation_runs
        from chapter4_qolu.hparam_sensitivity import perturb, resolve_hyperparameters
        from chapter4_qolu.hparam_worker import execute_hparam_run

        item = perturb(resolve_hyperparameters(["observation_window"])[0], +1)
        spec = next(
            entry
            for entry in perturbation_runs(tiny_settings, item)
            if entry.method == "qolu_buia"
        )
        record = execute_hparam_run(spec)
        assert record["env_settings"]["observation_window"] == 11


class TestRendering:
    """The report renders from the payload alone, with nothing on disk."""

    def _summary(self) -> dict[str, Any]:
        methods = {
            "qolu_buia": {
                "mean": 0.8,
                "sd": 0.02,
                "se": 0.01,
                "n": 3,
                "values": [0.79, 0.8, 0.81],
                "seeds": [0, 1, 2],
                "baseline_mean": 0.82,
                "delta": -0.02,
                "borrowed_from_baseline": False,
                "paired": {"p_value": 0.2, "significant": False, "possible": True},
            },
            "mitm": {
                "mean": 0.5,
                "sd": 0.02,
                "se": 0.01,
                "n": 3,
                "values": [0.49, 0.5, 0.51],
                "seeds": [0, 1, 2],
                "baseline_mean": 0.5,
                "delta": 0.0,
                "borrowed_from_baseline": True,
                "paired": {"p_value": 1.0, "significant": False, "possible": True},
            },
        }
        metrics = {
            metric: {
                "methods": methods,
                "ranking": ["qolu_buia", "mitm"],
                "baseline_ranking": ["qolu_buia", "mitm"],
                "ranking_preserved": True,
                "kendall_tau": 1.0,
                "absolute": {
                    "ranking": ["qolu_buia", "mitm"],
                    "baseline_ranking": ["qolu_buia", "mitm"],
                    "ranking_preserved": True,
                    "kendall_tau": 1.0,
                },
            }
            for metric in PUBLISHED_METRICS
        }
        return {
            "generated": "2026-09-29T00:00:00+00:00",
            "tag": "demo",
            "settings": {
                "raster": "lcm.tif",
                "fraction": 0.2,
                "iterations": 1000,
                "seeds": [0, 1, 2],
                "scenarios": [1],
                "methods": ["qolu_buia", "mitm"],
                "configuration": {"description": "6x6 grid"},
                "window": "E402_N110_6x6",
                "parallel": 1,
            },
            "hyperparameters": [
                {
                    "name": "gamma",
                    "target": "qolu",
                    "baseline": 0.99,
                    "source": "published",
                    "note": "Table 4.1 states gamma = 0.99.",
                    "minus": {"value": 0.792, "rounded": False, "clamped": False},
                    "plus": {"value": 1.0, "rounded": False, "clamped": True},
                }
            ],
            "perturbations": [
                {
                    "label": "gamma_plus",
                    "hyperparameter": "gamma",
                    "direction": "plus",
                    "target": "qolu",
                    "baseline": 0.99,
                    "requested": 1.188,
                    "value": 1.0,
                    "applied_fraction": 0.0101,
                    "rounded": False,
                    "clamped": True,
                    "no_op": False,
                    "description": "`gamma` +20 %: 0.99 -> 1.0",
                    "methods_rerun": ["qolu_buia"],
                    "n_records": 3,
                    "complete": True,
                    "scenarios": [{"scenario": 1, "metrics": metrics}],
                }
            ],
            "overall": {
                "n_perturbations": 1,
                "n_effective": 1,
                "n_no_op": 0,
                "n_scenarios": 1,
                "metrics": {
                    metric: {
                        "preserved": 1,
                        "total": 1,
                        "share": 1.0,
                        "mean_kendall_tau": 1.0,
                        "cells_preserved": 1,
                        "n_cells": 1,
                        "absolute": {
                            "preserved": 1,
                            "total": 1,
                            "mean_kendall_tau": 1.0,
                        },
                    }
                    for metric in PUBLISHED_METRICS
                },
                "all_metrics_preserved": 1,
            },
        }

    def test_the_report_is_markdown_with_the_expected_sections(self) -> None:
        text = render_hparam_report(self._summary())
        assert text.startswith("# Hyper-parameter sensitivity")
        assert "## What the thesis promised" in text
        assert "## The perturbed hyper-parameters" in text
        assert "## Does the method ranking survive?" in text
        assert "## Per-perturbation detail" in text
        assert "## Method" in text

    def test_a_clamped_perturbation_is_flagged_in_the_table(self) -> None:
        text = render_hparam_report(self._summary())
        assert "clamped" in text

    def test_the_headline_counts_appear(self) -> None:
        text = render_hparam_report(self._summary())
        assert "1 / 1" in text

    def test_a_missing_payload_still_renders(self) -> None:
        text = render_hparam_report({"tag": "empty"})
        assert "Hyper-parameter sensitivity" in text

    def test_the_csv_carries_the_perturbation_columns(self) -> None:
        records = [
            {
                "method": "qolu_buia",
                "scenario": 1,
                "seed": 0,
                "configuration": {"label": "6x6_f1.0_cap2"},
                "final_metrics": {"land_use": 0.1, "woodland": 0.2, "urban": 0.3},
                "final_metrics_absolute": {"land_use": 0.4},
                "total_conversions": 5,
                "runtime_seconds": 1.5,
                "hyperparameter": {
                    "label": "gamma_plus",
                    "hyperparameter": "gamma",
                    "direction": "plus",
                    "baseline": 0.99,
                    "value": 1.0,
                    "requested": 1.188,
                    "rounded": False,
                    "clamped": True,
                },
            }
        ]
        text = render_hparam_runs_csv(records)
        header, row = text.strip().splitlines()
        assert header == ",".join(HPARAM_RUN_COLUMNS)
        assert row.startswith("gamma_plus,gamma,plus,0.99,1.0,")

    def test_a_baseline_record_is_labelled_baseline(self) -> None:
        text = render_hparam_runs_csv(
            [
                {
                    "method": "mitm",
                    "scenario": 1,
                    "seed": 0,
                    "configuration": {"label": "6x6_f1.0_cap2"},
                    "final_metrics": {"land_use": 0.1},
                    "hyperparameter": None,
                }
            ]
        )
        assert text.strip().splitlines()[1].startswith(f"{BASELINE_TAG},,,,,")


class TestCommandLine:
    def test_the_parser_offers_the_house_options(self) -> None:
        parser = _build_parser()
        options = {action.dest for action in parser._actions}
        for name in (
            "raster",
            "tag",
            "hyperparameters",
            "fraction",
            "methods",
            "scenarios",
            "seeds",
            "iterations",
            "config",
            "window",
            "parallel",
            "resume",
            "report_only",
            "dry_run",
            "output_dir",
        ):
            assert name in options

    def test_help_mentions_the_hyperparameters(self) -> None:
        text = _build_parser().format_help()
        assert "learning_rate" in text
        assert "--dry-run" in text
        assert "--resume" in text
        assert "--report-only" in text

    def test_the_configuration_under_test_is_parsed(self, tmp_path: Path) -> None:
        raster = tmp_path / "r.tif"
        raster.write_bytes(b"")  # only the path is resolved, never opened
        arguments = _build_parser().parse_args(
            [
                "--raster",
                str(raster),
                "--config",
                "30:0.5:4",
                "--seeds",
                "0",
                "1",
                "2",
                "3",
                "--hyperparameters",
                "gamma",
                "--fraction",
                "0.1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        settings = _settings_from(arguments)
        assert settings.scale.label == "30x30_f0.5_cap4"
        assert settings.seeds == (0, 1, 2, 3)
        assert settings.hyperparameter_names == ("gamma",)
        assert settings.fraction == pytest.approx(0.1)

    def test_report_only_needs_no_raster(
        self, tiny_settings: HParamSettings, summary: dict[str, Any]
    ) -> None:
        code = main(
            [
                "--tag",
                "tiny",
                "--report-only",
                "--methods",
                "qolu_buia",
                "mitm",
                "--hyperparameters",
                "gamma",
                "samples_per_iteration",
                "--config",
                "6:1.0:2",
                "--window",
                "E402_N110_6x6",
                "--iterations",
                "3",
                "--stakeholders",
                "1",
                "--output-dir",
                str(hparam_paths(tiny_settings).root.parent),
            ]
        )
        assert code == 0
        assert hparam_paths(tiny_settings).report.is_file()

    def test_a_dry_run_from_the_command_line_writes_nothing(
        self, tiny_class_raster: Path, tmp_path: Path
    ) -> None:
        code = main(
            [
                "--raster",
                str(tiny_class_raster),
                "--tag",
                "plan",
                "--dry-run",
                "--config",
                "6:1.0:2",
                "--output-dir",
                str(tmp_path / "results"),
            ]
        )
        assert code == 0
        assert not (tmp_path / "results" / "hparam_plan").exists()
