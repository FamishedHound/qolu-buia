"""Every run record also carries its initial and final land-cover grids."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.reporting.paths import parse_scale_config
from chapter4_qolu.run_record import (
    GRID_NO_DATA,
    class_grid,
    decode_class_grid,
    encode_class_grid,
)
from chapter4_qolu.tests.raster_fixtures import write_geotiff


@pytest.fixture()
def tiny_class_raster(tmp_path: Path) -> Path:
    rng = np.random.default_rng(4242)
    codes = rng.integers(1, 11, size=(16, 16), dtype=np.uint8)
    return write_geotiff(
        tmp_path / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        codes,
        origin_e_m=400_000.0,
        origin_n_m=120_000.0,
    )


class TestEncoding:
    def test_a_grid_round_trips(self) -> None:
        values = np.array([[0, 3, 9], [-1, 10, 2]], dtype=np.int64)
        encoded = encode_class_grid(values)
        restored = decode_class_grid(encoded)
        assert restored.tolist() == values.tolist()

    def test_the_encoding_is_one_string_per_row(self) -> None:
        values = np.array([[0, 3], [4, 5]], dtype=np.int64)
        encoded = encode_class_grid(values)
        assert encoded["shape"] == [2, 2]
        assert encoded["rows"] == ["0,3", "4,5"]

    def test_the_encoding_stays_small(self) -> None:
        values = np.zeros((50, 50), dtype=np.int64)
        text = json.dumps(encode_class_grid(values), indent=2)
        assert len(text) < 8_000

    def test_decoding_rejects_a_ragged_payload(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            decode_class_grid({"shape": [2, 3], "rows": ["0,1", "2,3"]})

    def test_no_data_is_the_documented_sentinel(self) -> None:
        assert GRID_NO_DATA == -1


class TestRecordedGrids:
    def test_a_finished_run_stores_both_grids(
        self, tiny_class_raster: Path, tmp_path: Path
    ) -> None:
        from chapter4_qolu.scenario_worker import RunSpec, execute_run

        spec = RunSpec(
            raster=tiny_class_raster,
            method="mitm",
            scenario=1,
            seed=0,
            scale=parse_scale_config("8:1.0:10"),
            output_path=tmp_path / "mitm_seed0.json",
            iterations=5,
            stakeholders=1,
        )
        record = execute_run(spec)
        initial = decode_class_grid(record["initial_grid"])
        final = decode_class_grid(record["final_grid"])
        assert initial.shape == (8, 8)
        assert final.shape == (8, 8)
        # MitM converts parcels, so the final grid must differ somewhere.
        assert not np.array_equal(initial, final)

    def test_the_values_are_land_cover_band_indices(
        self, tiny_class_raster: Path, tmp_path: Path
    ) -> None:
        from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES
        from chapter4_qolu.scenario_worker import RunSpec, execute_run

        spec = RunSpec(
            raster=tiny_class_raster,
            method="mitm",
            scenario=1,
            seed=0,
            scale=parse_scale_config("8:1.0:10"),
            output_path=tmp_path / "mitm_seed0.json",
            iterations=3,
            stakeholders=1,
        )
        record = execute_run(spec)
        values = decode_class_grid(record["final_grid"])
        assert values.min() >= GRID_NO_DATA
        assert values.max() < N_AGGREGATE_CLASSES

    def test_class_grid_marks_empty_parcels_as_no_data(self) -> None:
        from chapter4_qolu.env.parcel_grid import ParcelGrid

        grid = ParcelGrid.empty(3, 3)
        values = class_grid(grid)
        assert set(np.unique(values).tolist()) == {GRID_NO_DATA}


class TestBackwardCompatibility:
    def test_a_record_without_grids_still_aggregates(self) -> None:
        from chapter4_qolu.reporting.aggregate import aggregate_records

        record = {
            "method": "mitm",
            "scenario": 1,
            "seed": 0,
            "iterations": 10,
            "configuration": {"label": "8x8_f1.0_cap10"},
            "final_metrics": {"land_use": 0.5, "woodland": 0.5, "urban": 0.5},
            "final_metrics_absolute": {
                "land_use": 0.5,
                "woodland": 0.5,
                "urban": 0.5,
            },
        }
        summary = aggregate_records([record])
        assert summary["n_runs"] == 1

    def test_the_figure_loader_skips_records_without_grids(self) -> None:
        from chapter4_qolu.figures.data import grids_by_method

        assert grids_by_method([{"method": "mitm"}]) == {}
