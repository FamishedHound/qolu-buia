"""Tests for the Southeast-England window grid used by the window scan."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.data.lcm_loader import load_lcm_raster
from chapter4_qolu.tests.raster_fixtures import write_geotiff
from chapter4_qolu.window_scan import (
    MAX_SEA_FRACTION,
    SOUTH_EAST_ENGLAND,
    WINDOW_SIZE_KM,
    WINDOW_STRIDE_KM,
    BoundingBoxKm,
    ScanWindow,
    candidate_windows,
    retain_windows,
    window_composition,
    window_origins,
)

#: The origin lists the scan documents; kept here so a change is visible.
EXPECTED_EASTINGS = (420, 445, 470, 495, 520, 545, 570, 590)
EXPECTED_NORTHINGS = (90, 115, 140, 165, 190)


class TestWindowOrigins:
    def test_walks_the_box_at_the_stride(self) -> None:
        assert window_origins(0, 100, 50, 25) == (0, 25, 50)

    def test_adds_a_flush_origin_when_a_wide_remainder_is_left(self) -> None:
        # 0..120 at stride 25 covers 0..100; the 20 km remainder is more than
        # half a stride, so a final window flush with the east edge is added.
        assert window_origins(0, 120, 50, 25) == (0, 25, 50, 70)

    def test_ignores_a_remainder_narrower_than_half_a_stride(self) -> None:
        assert window_origins(0, 110, 50, 25) == (0, 25, 50)

    def test_rejects_a_box_smaller_than_the_window(self) -> None:
        with pytest.raises(ValueError):
            window_origins(0, 40, 50, 25)

    def test_rejects_a_non_positive_stride(self) -> None:
        with pytest.raises(ValueError):
            window_origins(0, 100, 50, 0)


class TestCandidateWindows:
    def test_south_east_box_is_the_documented_one(self) -> None:
        assert SOUTH_EAST_ENGLAND == BoundingBoxKm(
            east_min=420, east_max=640, north_min=90, north_max=250
        )
        assert WINDOW_SIZE_KM == 50
        assert WINDOW_STRIDE_KM == 25
        assert MAX_SEA_FRACTION == pytest.approx(0.30)

    def test_produces_forty_windows_on_the_documented_origins(self) -> None:
        windows = candidate_windows()
        assert len(windows) == 40
        assert tuple(sorted({w.easting_km for w in windows})) == EXPECTED_EASTINGS
        assert tuple(sorted({w.northing_km for w in windows})) == EXPECTED_NORTHINGS

    def test_every_window_stays_inside_the_bounding_box(self) -> None:
        box = SOUTH_EAST_ENGLAND
        for window in candidate_windows():
            assert box.east_min <= window.easting_km
            assert window.easting_km + window.size_km <= box.east_max
            assert box.north_min <= window.northing_km
            assert window.northing_km + window.size_km <= box.north_max

    def test_windows_are_ordered_north_west_first(self) -> None:
        windows = candidate_windows()
        assert windows[0] == ScanWindow(easting_km=420, northing_km=190)
        assert windows[-1] == ScanWindow(easting_km=590, northing_km=90)


class TestScanWindowGeometry:
    def test_label_and_description_name_the_south_west_corner(self) -> None:
        window = ScanWindow(easting_km=495, northing_km=140)
        assert window.label == "E495_N140"
        assert "495-545" in window.description
        assert "140-190" in window.description

    def test_top_left_is_the_north_west_corner(self) -> None:
        window = ScanWindow(easting_km=495, northing_km=140)
        assert window.bng_top_left_km == (495, 190)

    def test_record_round_trips_through_json(self) -> None:
        import json

        record = ScanWindow(easting_km=420, northing_km=90).as_record()
        assert json.loads(json.dumps(record)) == record


@pytest.fixture()
def coastal_raster(tmp_path: Path) -> Path:
    """A 10 x 10 km class raster whose southern half is sea."""
    codes = np.full((10, 10), 4, dtype=np.uint8)
    codes[5:, :] = 0  # sea
    codes[0, :] = 10  # built-up strip along the north edge
    return write_geotiff(
        tmp_path / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        codes,
        origin_e_m=400_000.0,
        origin_n_m=110_000.0,
    )


class TestWindowComposition:
    def test_counts_sea_and_class_shares(self, coastal_raster: Path) -> None:
        raster = load_lcm_raster(coastal_raster)
        window = ScanWindow(easting_km=400, northing_km=100, size_km=10)
        composition = window_composition(raster, window)
        assert composition.n_cells == 100
        assert composition.sea_fraction == pytest.approx(0.5)
        assert composition.class_shares[4] == pytest.approx(0.4)
        assert composition.class_shares[10] == pytest.approx(0.1)
        assert 7 not in composition.class_shares

    def test_land_shares_and_sea_fraction_sum_to_one(
        self, coastal_raster: Path
    ) -> None:
        raster = load_lcm_raster(coastal_raster)
        window = ScanWindow(easting_km=400, northing_km=100, size_km=10)
        composition = window_composition(raster, window)
        total = composition.sea_fraction + sum(composition.class_shares.values())
        assert total == pytest.approx(1.0)

    def test_composition_record_is_json_serialisable(
        self, coastal_raster: Path
    ) -> None:
        import json

        raster = load_lcm_raster(coastal_raster)
        window = ScanWindow(easting_km=400, northing_km=100, size_km=10)
        record = window_composition(raster, window).as_record()
        assert json.loads(json.dumps(record)) == record


class TestSeaFilter:
    def test_drops_windows_above_the_sea_threshold(self, coastal_raster: Path) -> None:
        raster = load_lcm_raster(coastal_raster)
        windows = (
            ScanWindow(easting_km=400, northing_km=105, size_km=5),  # all land
            ScanWindow(easting_km=400, northing_km=100, size_km=10),  # half sea
        )
        kept, dropped = retain_windows(raster, windows, max_sea_fraction=0.30)
        assert [item.window for item in kept] == [windows[0]]
        assert [item.window for item in dropped] == [windows[1]]

    def test_a_window_exactly_at_the_threshold_is_kept(
        self, coastal_raster: Path
    ) -> None:
        raster = load_lcm_raster(coastal_raster)
        windows = (ScanWindow(easting_km=400, northing_km=100, size_km=10),)
        kept, dropped = retain_windows(raster, windows, max_sea_fraction=0.5)
        assert len(kept) == 1
        assert not dropped
