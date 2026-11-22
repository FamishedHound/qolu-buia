"""Scoring a record on ``c_align`` from the grids it stored.

A record that does not carry ``c_align`` stores its initial and final
dominant-class grids, which is everything the metric reads.  These tests pin
that scoring it from them is exact, never mutates a record, never invents a
value, and that the aggregation uses it and says so.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.env.class_grids import encode_class_grid
from chapter4_qolu.env.metrics import c_align_from_classes
from chapter4_qolu.reporting.aggregate import aggregate_records
from chapter4_qolu.reporting.derived import (
    FROM_STORED_GRIDS,
    MISSING,
    RECORDED,
    derived_metric_counts,
    metric_source,
    with_derived_metrics,
)
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.scenario_worker import RunSpec, execute_run
from chapter4_qolu.tests.raster_fixtures import write_geotiff
from chapter4_qolu.tests.test_reporting_aggregate import make_record

BEFORE = np.array([[0, 0, 0], [0, 2, 0], [3, 0, 3]], dtype=np.int64)
AFTER = np.where(BEFORE == 0, 9, BEFORE)


def legacy_record(method: str = "ltos", seed: int = 0) -> dict:
    """A record that stores its grids but does not carry ``c_align``."""
    record = make_record(method, seed, {"land_use": 0.6})
    for view in ("final_metrics", "final_metrics_absolute"):
        record[view].pop("c_align", None)
    record["initial_grid"] = encode_class_grid(BEFORE)
    record["final_grid"] = encode_class_grid(AFTER)
    return record


class TestScoringOneRecord:
    def test_a_legacy_record_is_scored_from_its_grids(self) -> None:
        derived = with_derived_metrics(legacy_record())
        expected = c_align_from_classes(BEFORE, AFTER)
        assert derived["final_metrics"]["c_align"] == expected
        assert derived["final_metrics_absolute"]["c_align"] == expected
        assert derived["derived_metrics"] == {"c_align": FROM_STORED_GRIDS}

    def test_the_input_is_left_untouched(self) -> None:
        record = legacy_record()
        snapshot = copy.deepcopy(record)
        with_derived_metrics(record)
        assert record == snapshot

    def test_a_recorded_value_is_kept(self) -> None:
        record = legacy_record()
        record["final_metrics"]["c_align"] = 0.123
        derived = with_derived_metrics(record)
        assert derived["final_metrics"]["c_align"] == 0.123
        assert "derived_metrics" not in derived

    def test_without_grids_nothing_is_invented(self) -> None:
        record = legacy_record()
        del record["initial_grid"]
        derived = with_derived_metrics(record)
        assert "c_align" not in derived["final_metrics"]
        assert metric_source(record) == MISSING

    def test_sources_are_counted(self) -> None:
        recorded = legacy_record()
        recorded["final_metrics"]["c_align"] = 0.2
        no_grids = legacy_record()
        del no_grids["final_grid"]
        counts = derived_metric_counts([recorded, legacy_record(), no_grids])
        assert counts == {"c_align": {RECORDED: 1, FROM_STORED_GRIDS: 1, MISSING: 1}}


class TestTheAggregationUsesIt:
    def test_legacy_records_get_a_c_align_column(self) -> None:
        records = [
            legacy_record(method, seed)
            for method in ("ltos", "qolu_buia")
            for seed in range(3)
        ]
        summary = aggregate_records(records)
        block = summary["configurations"][0]["scenarios"][0]["methods"]["ltos"]
        expected = c_align_from_classes(BEFORE, AFTER)
        assert block["raw"]["c_align"]["mean"] == pytest.approx(expected)
        assert block["absolute"]["c_align"]["mean"] == pytest.approx(expected)
        assert summary["derived_metrics"]["c_align"][FROM_STORED_GRIDS] == 6

    def test_the_records_handed_in_are_not_modified(self) -> None:
        records = [legacy_record(method, 0) for method in ("ltos", "qolu_buia")]
        snapshot = copy.deepcopy(records)
        aggregate_records(records)
        assert records == snapshot


@pytest.fixture(scope="module")
def tiny_raster(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(424242)
    return write_geotiff(
        tmp_path_factory.mktemp("derived_raster")
        / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        rng.integers(1, 11, size=(16, 16), dtype=np.uint8),
        origin_e_m=400_000.0,
        origin_n_m=124_000.0,
    )


class TestTheTwoRoutesAgreeOnARealRun:
    def test_the_in_run_value_equals_the_stored_grid_value(
        self, tiny_raster: Path, tmp_path: Path
    ) -> None:
        record = execute_run(
            RunSpec(
                raster=tiny_raster,
                method="mitm",
                scenario=1,
                seed=0,
                scale=ScaleConfig(
                    grid_size=10, conversion_fraction=1.0, max_conversions=1
                ),
                output_path=tmp_path / "mitm.json",
                iterations=15,
                window_origin=(0, 0),
            )
        )
        in_run = record["final_metrics"]["c_align"]
        assert in_run > 0.0
        stripped = copy.deepcopy(record)
        for view in ("final_metrics", "final_metrics_absolute"):
            stripped[view].pop("c_align")
        assert with_derived_metrics(stripped)["final_metrics"]["c_align"] == in_run
