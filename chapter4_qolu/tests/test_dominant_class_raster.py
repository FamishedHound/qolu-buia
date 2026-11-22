"""Tests for the dominant-aggregate-class raster path of the LCM loader.

The paper's 1 km percentage-cover product has ten bands; UKCEH also publish a
single-band *dominant aggregate class* product for the whole of GB, which is
what the Southeast window scan runs on.  These tests use synthetic class
rasters only - no fixture data is needed, so they run everywhere rasterio does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.data.lcm_loader import (
    DOMINANT_CLASS_NO_DATA,
    LCMRaster,
    bng_origin_to_index,
    load_lcm_raster,
    select_region_of_interest,
)
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES
from chapter4_qolu.tests.raster_fixtures import (
    SYNTHETIC_ORIGIN_E_M,
    SYNTHETIC_ORIGIN_N_M,
    write_geotiff,
)


@pytest.fixture()
def class_codes() -> np.ndarray:
    """A 6 x 8 patch of class codes with sea (0) along the bottom row."""
    codes = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 1, 2, 3, 4],
            [10, 10, 10, 4, 4, 4, 1, 1],
            [3, 3, 3, 3, 10, 10, 10, 10],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    return codes


@pytest.fixture()
def class_raster_path(tmp_path: Path, class_codes: np.ndarray) -> Path:
    name = "lcm2015_gb_1km_dominant_aggregate_class.tif"
    return write_geotiff(tmp_path / name, class_codes)


@pytest.fixture()
def percentage_raster_path(tmp_path: Path) -> Path:
    """A ten-band percentage product, so the existing path can be re-checked."""
    percentages = np.zeros((N_AGGREGATE_CLASSES, 4, 4), dtype=np.uint8)
    percentages[0] = 60
    percentages[3] = 40
    name = "lcm2015_gb_1km_percent_cover_aggregate_class.tif"
    return write_geotiff(tmp_path / name, percentages)


class TestDominantClassLoading:
    def test_single_band_raster_becomes_one_hot_cover(
        self, class_raster_path: Path, class_codes: np.ndarray
    ) -> None:
        raster = load_lcm_raster(class_raster_path)
        assert raster.fractions.shape == (N_AGGREGATE_CLASSES, *class_codes.shape)
        for row in range(class_codes.shape[0]):
            for col in range(class_codes.shape[1]):
                code = int(class_codes[row, col])
                column = raster.fractions[:, row, col]
                if code == DOMINANT_CLASS_NO_DATA:
                    assert float(column.sum()) == pytest.approx(0.0)
                else:
                    assert float(column[code - 1]) == pytest.approx(1.0)
                    assert float(column.sum()) == pytest.approx(1.0)

    def test_sea_cells_carry_no_land_at_all(self, class_raster_path: Path) -> None:
        raster = load_lcm_raster(class_raster_path)
        grid = select_region_of_interest(raster, grid_size=6, origin=(0, 0))
        land = grid.land_mask()
        assert not land[5].any()
        assert land[:5].all()

    def test_band_order_follows_land_cover_module(
        self, class_raster_path: Path
    ) -> None:
        raster = load_lcm_raster(class_raster_path)
        # Code 10 (built-up) has to land in the last band, code 1 in the first.
        assert float(raster.fractions[9, 1, 1]) == pytest.approx(1.0)
        assert float(raster.fractions[0, 0, 0]) == pytest.approx(1.0)

    def test_rejects_out_of_range_class_codes(self, tmp_path: Path) -> None:
        codes = np.array([[1, 2], [3, 77]], dtype=np.uint8)
        path = write_geotiff(tmp_path / "broken_dominant_class.tif", codes)
        with pytest.raises(ValueError, match="class code"):
            load_lcm_raster(path)

    def test_rejects_a_band_count_that_is_neither_one_nor_ten(
        self, tmp_path: Path
    ) -> None:
        values = np.zeros((4, 3, 3), dtype=np.uint8)
        path = write_geotiff(tmp_path / "four_band.tif", values)
        with pytest.raises(ValueError, match="bands"):
            load_lcm_raster(path)

    def test_percentage_product_path_is_unchanged(
        self, percentage_raster_path: Path
    ) -> None:
        raster = load_lcm_raster(percentage_raster_path)
        assert raster.fractions.shape == (N_AGGREGATE_CLASSES, 4, 4)
        assert float(raster.fractions[0, 0, 0]) == pytest.approx(0.6)
        assert float(raster.fractions[3, 0, 0]) == pytest.approx(0.4)


class TestGeoreferencing:
    def test_exposes_the_top_left_corner_in_crs_units(
        self, class_raster_path: Path
    ) -> None:
        raster = load_lcm_raster(class_raster_path)
        assert raster.origin_xy is not None
        easting, northing = raster.origin_xy
        assert easting == pytest.approx(SYNTHETIC_ORIGIN_E_M)
        assert northing == pytest.approx(SYNTHETIC_ORIGIN_N_M)

    def test_aggregating_pixels_keeps_the_top_left_corner(
        self, class_raster_path: Path
    ) -> None:
        raster = load_lcm_raster(class_raster_path)
        aggregated = raster.aggregate_pixels(block=2)
        assert aggregated.origin_xy == raster.origin_xy

    def test_bng_origin_maps_onto_row_and_column(self, class_raster_path: Path) -> None:
        raster = load_lcm_raster(class_raster_path)
        # The raster's own north-west corner is (400 km E, 200 km N).
        assert bng_origin_to_index(raster, 400.0, 200.0) == (0, 0)
        # One kilometre east and two south.
        assert bng_origin_to_index(raster, 401.0, 198.0) == (2, 1)

    def test_bng_origin_selects_the_same_window_as_row_col(
        self, class_raster_path: Path
    ) -> None:
        raster = load_lcm_raster(class_raster_path)
        by_index = select_region_of_interest(raster, grid_size=3, origin=(1, 2))
        by_bng = select_region_of_interest(
            raster, grid_size=3, bng_origin_km=(402.0, 199.0)
        )
        np.testing.assert_allclose(by_bng.composition, by_index.composition)

    def test_rejects_a_bng_origin_off_the_grid(self, class_raster_path: Path) -> None:
        raster = load_lcm_raster(class_raster_path)
        with pytest.raises(ValueError, match="parcel boundary"):
            bng_origin_to_index(raster, 400.4, 200.0)

    def test_rejects_a_bng_origin_outside_the_raster(
        self, class_raster_path: Path
    ) -> None:
        raster = load_lcm_raster(class_raster_path)
        with pytest.raises(ValueError):
            select_region_of_interest(raster, grid_size=3, bng_origin_km=(300.0, 200.0))

    def test_rejects_both_origins_at_once(self, class_raster_path: Path) -> None:
        raster = load_lcm_raster(class_raster_path)
        with pytest.raises(ValueError, match="not both"):
            select_region_of_interest(
                raster, grid_size=3, origin=(0, 0), bng_origin_km=(400.0, 200.0)
            )

    def test_bng_origin_needs_a_georeferenced_raster(self) -> None:
        raster = LCMRaster(
            fractions=np.zeros((N_AGGREGATE_CLASSES, 4, 4), dtype=np.float32),
            path=Path("synthetic.tif"),
            pixel_size_m=1000.0,
        )
        with pytest.raises(ValueError, match="georeferen"):
            select_region_of_interest(raster, grid_size=2, bng_origin_km=(400.0, 200.0))
