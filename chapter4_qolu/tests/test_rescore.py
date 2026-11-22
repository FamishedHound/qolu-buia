"""Offline re-scoring: the saturation distance is chosen without re-running."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.reporting.paths import parse_scale_config
from chapter4_qolu.rescore import (
    CANDIDATE_DISTANCES_KM,
    fit_max_distance,
    grid_from_classes,
    rescore_record,
    rescore_records,
)
from chapter4_qolu.run_record import GRID_NO_DATA, decode_class_grid
from chapter4_qolu.tests.raster_fixtures import write_geotiff


@pytest.fixture()
def tiny_class_raster(tmp_path: Path) -> Path:
    rng = np.random.default_rng(31415)
    codes = rng.integers(1, 11, size=(20, 20), dtype=np.uint8)
    return write_geotiff(
        tmp_path / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        codes,
        origin_e_m=400_000.0,
        origin_n_m=120_000.0,
    )


@pytest.fixture()
def finished_record(tiny_class_raster: Path, tmp_path: Path) -> dict:
    from chapter4_qolu.scenario_worker import RunSpec, execute_run

    spec = RunSpec(
        raster=tiny_class_raster,
        method="mitm",
        scenario=1,
        seed=0,
        scale=parse_scale_config("12:1.0:10"),
        output_path=tmp_path / "mitm_seed0.json",
        iterations=20,
        stakeholders=1,
        window_origin=(0, 0),
        max_distance_km=10.0,
    )
    return execute_run(spec)


class TestOneHotReconstruction:
    def test_a_class_grid_becomes_a_one_hot_parcel_grid(self) -> None:
        values = np.array([[0, 3], [9, GRID_NO_DATA]], dtype=np.int64)
        grid = grid_from_classes(values)
        assert grid.shape == (2, 2)
        assert grid.composition[0, 0, 0] == pytest.approx(1.0)
        assert grid.composition[3, 0, 1] == pytest.approx(1.0)
        assert grid.composition[9, 1, 0] == pytest.approx(1.0)
        # No-data stays an all-zero column, i.e. what land_mask already drops.
        assert float(grid.composition[:, 1, 1].sum()) == pytest.approx(0.0)
        assert grid.land_mask().tolist() == [[True, True], [True, False]]

    def test_the_reconstruction_is_two_dimensional_only(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            grid_from_classes(np.zeros((2, 2, 2), dtype=np.int64))


class TestExactness:
    def test_recomputing_at_ten_km_reproduces_the_records_own_absolute_view(
        self, finished_record: dict
    ) -> None:
        recomputed = rescore_record(finished_record, 10.0)
        assert recomputed is not None
        stored = finished_record["final_metrics_absolute"]
        for metric in ("land_use", "woodland", "urban"):
            assert recomputed[metric] == pytest.approx(stored[metric], abs=1e-12)

    def test_recomputing_without_a_distance_reproduces_the_raw_view(
        self, finished_record: dict
    ) -> None:
        recomputed = rescore_record(finished_record, None)
        assert recomputed is not None
        stored = finished_record["final_metrics"]
        for metric in ("land_use", "woodland", "urban"):
            assert recomputed[metric] == pytest.approx(stored[metric], abs=1e-12)

    def test_the_stored_grids_round_trip(self, finished_record: dict) -> None:
        initial = decode_class_grid(finished_record["initial_grid"])
        final = decode_class_grid(finished_record["final_grid"])
        assert initial.shape == final.shape == (12, 12)

    def test_a_record_without_grids_cannot_be_rescored(self) -> None:
        assert rescore_record({"method": "mitm"}, 10.0) is None


class TestDistanceSensitivity:
    def test_a_shorter_saturation_distance_never_lowers_a_proximity_score(
        self, finished_record: dict
    ) -> None:
        near = rescore_record(finished_record, 5.0)
        far = rescore_record(finished_record, 20.0)
        assert near is not None and far is not None
        # land_use is "distance to habitat, saturating at the limit", so a
        # smaller limit can only push it up; urban is 1 - that distance.
        assert near["land_use"] >= far["land_use"] - 1e-12
        assert near["urban"] <= far["urban"] + 1e-12

    def test_woodland_does_not_depend_on_the_distance(
        self, finished_record: dict
    ) -> None:
        near = rescore_record(finished_record, 5.0)
        far = rescore_record(finished_record, 20.0)
        assert near is not None and far is not None
        assert near["woodland"] == pytest.approx(far["woodland"])


class TestFitting:
    def _records(self) -> list[dict]:
        """Four synthetic methods over one scenario, with grids."""
        from chapter4_qolu.run_record import encode_class_grid

        rng = np.random.default_rng(7)
        initial = rng.integers(0, 10, size=(12, 12)).astype(np.int64)
        records = []
        for index, method in enumerate(("qolu_buia", "pmaddpg", "ltos", "mitm")):
            final = initial.copy()
            final[: index + 1, :] = 9
            records.append(
                {
                    "method": method,
                    "scenario": 1,
                    "seed": 0,
                    "initial_grid": encode_class_grid(initial),
                    "final_grid": encode_class_grid(final),
                }
            )
        return records

    def test_every_candidate_distance_is_scored(self) -> None:
        fits = fit_max_distance(self._records())
        assert len(fits) == len(CANDIDATE_DISTANCES_KM)
        assert {fit.max_distance_km for fit in fits} == set(CANDIDATE_DISTANCES_KM)

    def test_the_best_fit_comes_first(self) -> None:
        fits = fit_max_distance(self._records())
        maes = [fit.value_mae for fit in fits if fit.value_mae is not None]
        assert maes == sorted(maes)

    def test_a_fit_records_its_per_scenario_breakdown(self) -> None:
        fits = fit_max_distance(self._records())
        record = fits[0].as_record()
        assert record["max_distance_km"] in CANDIDATE_DISTANCES_KM
        assert "1" in record["per_scenario"]

    def test_records_without_grids_yield_no_fit(self) -> None:
        fits = fit_max_distance([{"method": "mitm", "scenario": 1}])
        assert all(fit.value_mae is None for fit in fits)

    def test_seeds_of_one_cell_are_averaged(self) -> None:
        from chapter4_qolu.run_record import encode_class_grid

        initial = np.zeros((6, 6), dtype=np.int64)
        records = []
        for seed in (0, 1):
            final = initial.copy()
            final[: seed + 1, :] = 9
            records.append(
                {
                    "method": "mitm",
                    "scenario": 2,
                    "seed": seed,
                    "initial_grid": encode_class_grid(initial),
                    "final_grid": encode_class_grid(final),
                }
            )
        values = rescore_records(records, 10.0)
        assert set(values) == {2}
        assert set(values[2]["woodland"]) == {"mitm"}

    def test_a_variant_is_filed_under_its_published_parent(self) -> None:
        from chapter4_qolu.run_record import encode_class_grid

        initial = np.zeros((6, 6), dtype=np.int64)
        final = initial.copy()
        final[0, :] = 9
        values = rescore_records(
            [
                {
                    "method": "mitm_learned",
                    "scenario": 1,
                    "seed": 0,
                    "initial_grid": encode_class_grid(initial),
                    "final_grid": encode_class_grid(final),
                }
            ],
            10.0,
        )
        assert set(values[1]["urban"]) == {"mitm"}
