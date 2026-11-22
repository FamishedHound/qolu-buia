"""Tests for the UKCEH LCM percentage-aggregate-class GeoTIFF loader.

These run against whatever raster the ``lcm_tif_path`` fixture resolves to; by
default that is the Northern-Ireland placeholder product, which is *not* the
study region used in the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.data.lcm_loader import (
    LCMRaster,
    is_placeholder_region,
    load_lcm_raster,
    select_region_of_interest,
)
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES


class TestPlaceholderRegionHeuristic:
    @pytest.mark.parametrize(
        "name",
        [
            "lcm2015_ni_1km_percent_cover_aggregate_class.tif",
            "LCM2015_NI_1km.tif",
            "lcm2015_northern_ireland_1km.tif",
        ],
    )
    def test_detects_northern_ireland_products(self, name: str) -> None:
        assert is_placeholder_region(Path(name))

    @pytest.mark.parametrize(
        "name",
        [
            "lcm2015_gb_1km_percent_cover_aggregate_class.tif",
            "southeast_uk_2015.tif",
        ],
    )
    def test_accepts_other_regions(self, name: str) -> None:
        assert not is_placeholder_region(Path(name))


class TestLoadLCMRaster:
    def test_loads_ten_aggregate_class_bands(self, lcm_tif_path: Path) -> None:
        raster = load_lcm_raster(lcm_tif_path)
        assert isinstance(raster, LCMRaster)
        assert raster.fractions.shape[0] == N_AGGREGATE_CLASSES
        assert raster.fractions.ndim == 3

    def test_percentages_are_converted_to_fractions(self, lcm_tif_path: Path) -> None:
        raster = load_lcm_raster(lcm_tif_path)
        assert raster.fractions.dtype == np.float32
        assert float(raster.fractions.min()) >= 0.0
        assert float(raster.fractions.max()) <= 1.0
        totals = raster.fractions.sum(axis=0)
        assert float(totals.max()) <= 1.0 + 1e-5

    def test_exposes_geospatial_metadata(self, lcm_tif_path: Path) -> None:
        raster = load_lcm_raster(lcm_tif_path)
        assert raster.pixel_size_m == pytest.approx(1000.0)
        assert raster.path == Path(lcm_tif_path)

    def test_warns_when_loading_placeholder_region_data(
        self, lcm_tif_path: Path
    ) -> None:
        if not is_placeholder_region(lcm_tif_path):
            pytest.skip("configured raster is not the NI placeholder")
        with pytest.warns(UserWarning, match="placeholder region"):
            load_lcm_raster(lcm_tif_path)

    def test_missing_file_raises_a_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_lcm_raster(tmp_path / "does_not_exist.tif")

    def test_backend_can_be_selected_explicitly(self, lcm_tif_path: Path) -> None:
        pytest.importorskip("rasterio")
        pytest.importorskip("tifffile")
        rasterio_raster = load_lcm_raster(lcm_tif_path, backend="rasterio")
        tifffile_raster = load_lcm_raster(lcm_tif_path, backend="tifffile")
        np.testing.assert_allclose(rasterio_raster.fractions, tifffile_raster.fractions)

    def test_unknown_backend_raises(self, lcm_tif_path: Path) -> None:
        with pytest.raises(ValueError):
            load_lcm_raster(lcm_tif_path, backend="gdal-direct")


class TestRegionOfInterest:
    def test_returns_a_square_parcel_grid_of_the_requested_size(
        self, lcm_tif_path: Path
    ) -> None:
        raster = load_lcm_raster(lcm_tif_path)
        grid = select_region_of_interest(raster, grid_size=50)
        assert isinstance(grid, ParcelGrid)
        assert grid.composition.shape == (N_AGGREGATE_CLASSES, 50, 50)

    def test_picks_a_window_with_land_in_it(self, lcm_tif_path: Path) -> None:
        raster = load_lcm_raster(lcm_tif_path)
        grid = select_region_of_interest(raster, grid_size=20)
        assert grid.land_mask().mean() > 0.9

    def test_origin_can_be_pinned_explicitly(self, lcm_tif_path: Path) -> None:
        raster = load_lcm_raster(lcm_tif_path)
        grid = select_region_of_interest(raster, grid_size=10, origin=(5, 7))
        np.testing.assert_allclose(
            grid.composition, raster.fractions[:, 5:15, 7:17], atol=1e-6
        )

    def test_rejects_a_grid_larger_than_the_raster(self, lcm_tif_path: Path) -> None:
        raster = load_lcm_raster(lcm_tif_path)
        with pytest.raises(ValueError):
            select_region_of_interest(raster, grid_size=10_000)

    def test_rejects_an_out_of_bounds_origin(self, lcm_tif_path: Path) -> None:
        raster = load_lcm_raster(lcm_tif_path)
        with pytest.raises(ValueError):
            select_region_of_interest(raster, grid_size=10, origin=(-1, 0))


class TestPixelAggregation:
    def test_aggregates_blocks_of_pixels_into_parcels(self) -> None:
        # 40 x 40 blocks of 25 m pixels are what the paper's Figure 1 describes;
        # the 1 km product used here is already at parcel resolution.
        fractions = np.zeros((N_AGGREGATE_CLASSES, 4, 4), dtype=np.float32)
        fractions[0, :2, :2] = 1.0
        fractions[1, :2, 2:] = 1.0
        raster = LCMRaster(
            fractions=fractions, path=Path("synthetic.tif"), pixel_size_m=25.0
        )
        aggregated = raster.aggregate_pixels(block=2)
        assert aggregated.fractions.shape == (N_AGGREGATE_CLASSES, 2, 2)
        assert aggregated.fractions[0, 0, 0] == pytest.approx(1.0)
        assert aggregated.fractions[1, 0, 1] == pytest.approx(1.0)
        assert aggregated.pixel_size_m == pytest.approx(50.0)

    def test_block_of_one_is_a_no_op(self) -> None:
        fractions = np.zeros((N_AGGREGATE_CLASSES, 2, 2), dtype=np.float32)
        fractions[3] = 0.5
        raster = LCMRaster(
            fractions=fractions, path=Path("synthetic.tif"), pixel_size_m=1000.0
        )
        np.testing.assert_allclose(
            raster.aggregate_pixels(block=1).fractions, fractions
        )

    def test_rejects_non_divisible_blocks(self) -> None:
        fractions = np.zeros((N_AGGREGATE_CLASSES, 3, 3), dtype=np.float32)
        raster = LCMRaster(
            fractions=fractions, path=Path("synthetic.tif"), pixel_size_m=25.0
        )
        with pytest.raises(ValueError):
            raster.aggregate_pixels(block=2)
