"""Tests for the campaign aggregation (summary.json)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from chapter4_qolu.reporting.aggregate import (
    ALL_METRICS,
    PUBLISHED_METRICS,
    REFERENCE_METHOD,
    aggregate_records,
    load_run_records,
    min_max_normalise,
    rank_methods,
)

_SCALE = {
    "label": "30x30_f1.0_cap10",
    "grid_size": 30,
    "conversion_fraction": 1.0,
    "max_conversions_per_iteration": 10,
}


def make_record(
    method: str,
    seed: int,
    metrics: dict[str, float],
    scenario: int = 1,
    absolute: dict[str, float] | None = None,
) -> dict[str, Any]:
    """A minimal per-run record in the shape ``run_scenarios`` writes."""
    full = {name: 0.5 for name in ALL_METRICS}
    full.update(metrics)
    absolute_full = dict(full) if absolute is None else {**full, **absolute}
    return {
        "method": method,
        "scenario": scenario,
        "seed": seed,
        "weights": {"land_use": 0.33, "woodland": 0.33, "urban": 0.33},
        "iterations": 50,
        "runtime_seconds": 1.0 + seed,
        "total_conversions": 100 + seed,
        "saturation_iteration": None,
        "configuration": dict(_SCALE),
        "absolute_max_distance_km": 10.0,
        "final_metrics": full,
        "final_metrics_absolute": absolute_full,
    }


def cell(summary: dict[str, Any], scenario: int = 1) -> dict[str, Any]:
    configuration = summary["configurations"][0]
    for entry in configuration["scenarios"]:
        if entry["scenario"] == scenario:
            return entry
    raise AssertionError(f"scenario {scenario} missing from the summary")


class TestRanking:
    def test_orders_methods_best_first(self) -> None:
        ranking = rank_methods({"mitm": 0.1, "qolu_buia": 0.9, "ltos": 0.5})
        assert ranking == ("qolu_buia", "ltos", "mitm")

    def test_breaks_ties_by_method_name_for_determinism(self) -> None:
        assert rank_methods({"b": 0.5, "a": 0.5}) == ("a", "b")

    def test_empty_input_gives_an_empty_ranking(self) -> None:
        assert rank_methods({}) == ()


class TestMinMaxNormalisation:
    def test_worst_is_zero_and_best_is_one(self) -> None:
        scaled = min_max_normalise({"a": 0.2, "b": 0.6, "c": 1.0})
        assert scaled["a"] == pytest.approx(0.0)
        assert scaled["c"] == pytest.approx(1.0)
        assert scaled["b"] == pytest.approx(0.5)

    def test_preserves_relative_spacing(self) -> None:
        scaled = min_max_normalise({"a": 10.0, "b": 12.5, "c": 20.0})
        assert scaled["b"] == pytest.approx((12.5 - 10.0) / 10.0)

    def test_a_flat_cell_maps_everything_to_one_half(self) -> None:
        scaled = min_max_normalise({"a": 0.4, "b": 0.4})
        assert scaled == {"a": 0.5, "b": 0.5}

    def test_a_single_method_maps_to_one_half(self) -> None:
        assert min_max_normalise({"only": 0.9}) == {"only": 0.5}


class TestAggregateRecords:
    @pytest.fixture()
    def records(self) -> list[dict[str, Any]]:
        values = {
            "qolu_buia": [0.80, 0.82, 0.84],
            "pmaddpg": [0.70, 0.72, 0.74],
            "ltos": [0.60, 0.62, 0.64],
            "mitm": [0.50, 0.52, 0.54],
        }
        return [
            make_record(
                method,
                seed,
                {"land_use": value, "woodland": value, "urban": value},
                absolute={"land_use": value * 2.0},
            )
            for method, series in values.items()
            for seed, value in enumerate(series)
        ]

    def test_summarises_every_metric_over_seeds(
        self, records: list[dict[str, Any]]
    ) -> None:
        summary = aggregate_records(records)
        entry = cell(summary)["methods"]["qolu_buia"]
        assert entry["n_seeds"] == 3
        assert entry["seeds"] == [0, 1, 2]
        assert set(entry["raw"]) == set(ALL_METRICS)
        assert entry["raw"]["land_use"]["mean"] == pytest.approx(0.82)
        assert entry["raw"]["land_use"]["sd"] == pytest.approx(0.02)
        assert entry["raw"]["land_use"]["se"] == pytest.approx(0.02 / math.sqrt(3))

    def test_keeps_the_absolute_view_separate_from_the_raw_view(
        self, records: list[dict[str, Any]]
    ) -> None:
        entry = cell(aggregate_records(records))["methods"]["qolu_buia"]
        assert entry["raw"]["land_use"]["mean"] == pytest.approx(0.82)
        assert entry["absolute"]["land_use"]["mean"] == pytest.approx(1.64)

    def test_min_max_view_spans_the_methods_of_the_cell(
        self, records: list[dict[str, Any]]
    ) -> None:
        methods = cell(aggregate_records(records))["methods"]
        assert methods["qolu_buia"]["minmax"]["land_use"]["mean"] == pytest.approx(1.0)
        assert methods["mitm"]["minmax"]["land_use"]["mean"] == pytest.approx(0.0)
        # 0.70 sits a third of the way from 0.52 to 0.82.
        assert methods["pmaddpg"]["minmax"]["land_use"]["mean"] == pytest.approx(
            (0.72 - 0.52) / (0.82 - 0.52)
        )

    def test_min_max_view_rescales_the_standard_error_by_the_same_span(
        self, records: list[dict[str, Any]]
    ) -> None:
        methods = cell(aggregate_records(records))["methods"]
        raw_se = methods["ltos"]["raw"]["land_use"]["se"]
        span = 0.82 - 0.52
        assert methods["ltos"]["minmax"]["land_use"]["se"] == pytest.approx(
            raw_se / span
        )

    def test_ranking_uses_the_raw_view(self, records: list[dict[str, Any]]) -> None:
        ranking = cell(aggregate_records(records))["ranking"]
        assert ranking["land_use"] == ["qolu_buia", "pmaddpg", "ltos", "mitm"]
        assert set(ranking) == set(ALL_METRICS)

    def test_pairs_every_baseline_against_the_reference_method(
        self, records: list[dict[str, Any]]
    ) -> None:
        tests = cell(aggregate_records(records))["tests"]
        assert set(tests) == {"pmaddpg", "ltos", "mitm"}
        assert set(tests["pmaddpg"]) == set(PUBLISHED_METRICS)
        entry = tests["pmaddpg"]["land_use"]
        # qolu - pmaddpg is a constant +0.10 across the three seeds.
        assert entry["possible"] is True
        assert entry["df"] == 2
        assert entry["mean_difference"] == pytest.approx(0.10)
        assert entry["p_value"] == pytest.approx(0.0)

    def test_reports_that_a_single_seed_cannot_be_tested(self) -> None:
        records = [
            make_record(method, 0, {"land_use": value})
            for method, value in (("qolu_buia", 0.8), ("mitm", 0.5))
        ]
        entry = cell(aggregate_records(records))["tests"]["mitm"]["land_use"]
        assert entry["possible"] is False
        assert entry["p_value"] is None
        assert "no paired t-test" in entry["note"]

    def test_only_pairs_seeds_both_methods_actually_ran(self) -> None:
        records = [
            make_record("qolu_buia", seed, {"land_use": 0.8}) for seed in (0, 1, 2, 3)
        ]
        records += [make_record("mitm", seed, {"land_use": 0.5}) for seed in (0, 1, 2)]
        entry = cell(aggregate_records(records))["tests"]["mitm"]["land_use"]
        assert entry["n"] == 3

    def test_records_the_reference_method_and_the_view_descriptions(
        self, records: list[dict[str, Any]]
    ) -> None:
        summary = aggregate_records(records)
        assert summary["reference_method"] == REFERENCE_METHOD == "qolu_buia"
        assert set(summary["views"]) == {"raw", "absolute", "minmax"}
        for description in summary["views"].values():
            assert isinstance(description, str) and description

    def test_groups_by_configuration_and_scenario(self) -> None:
        records = [
            make_record("qolu_buia", 0, {"land_use": 0.8}, scenario=scenario)
            for scenario in (1, 2)
        ]
        other = make_record("qolu_buia", 0, {"land_use": 0.4}, scenario=1)
        other["configuration"] = {
            "label": "50x50_f0.25_capnone",
            "grid_size": 50,
            "conversion_fraction": 0.25,
            "max_conversions_per_iteration": None,
        }
        summary = aggregate_records([*records, other])
        assert [entry["label"] for entry in summary["configurations"]] == [
            "30x30_f1.0_cap10",
            "50x50_f0.25_capnone",
        ]
        assert [
            entry["scenario"] for entry in summary["configurations"][0]["scenarios"]
        ]
        assert len(summary["configurations"][0]["scenarios"]) == 2
        assert len(summary["configurations"][1]["scenarios"]) == 1

    def test_carries_the_absolute_view_distance_and_any_metadata(
        self, records: list[dict[str, Any]]
    ) -> None:
        summary = aggregate_records(records, metadata={"tag": "smoke"})
        assert summary["absolute_max_distance_km"] == pytest.approx(10.0)
        assert summary["campaign"]["tag"] == "smoke"

    def test_is_json_serialisable(self, records: list[dict[str, Any]]) -> None:
        json.dumps(aggregate_records(records))

    def test_rejects_an_empty_campaign(self) -> None:
        with pytest.raises(ValueError):
            aggregate_records([])


class TestLoadRunRecords:
    def test_reads_every_json_under_the_campaign_directory(
        self, tmp_path: Path
    ) -> None:
        for scenario in (1, 2):
            directory = tmp_path / _SCALE["label"] / f"scenario{scenario}"
            directory.mkdir(parents=True)
            for method in ("qolu_buia", "mitm"):
                record = make_record(method, 0, {"land_use": 0.5}, scenario=scenario)
                (directory / f"{method}_seed0.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
        records = load_run_records(tmp_path)
        assert len(records) == 4
        assert {record["method"] for record in records} == {"qolu_buia", "mitm"}

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_run_records(tmp_path / "absent")
