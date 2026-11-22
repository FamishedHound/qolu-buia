"""Whole-region mode: a non-square South East England extent.

The paper's Figure 1 shows the *entire* South East England region at 1 km
resolution, while the text says "a 50 x 50 grid of parcels, as shown in Figure
1".  The two cannot both be true, so the environment has to be able to express
either; these tests pin the rectangular reading down.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.config import EnvConfig
from chapter4_qolu.data.lcm_loader import (
    load_lcm_raster,
    resolve_window_origin,
    select_region,
    select_region_of_interest,
)
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.reporting.paths import ScaleConfig, parse_scale_config
from chapter4_qolu.tests.raster_fixtures import write_geotiff
from chapter4_qolu.window_scan import (
    SOUTH_EAST_REGION,
    SOUTH_EAST_REGION_LABEL,
    ScanWindow,
    parse_window_label,
    region_window,
)


@pytest.fixture()
def wide_class_raster(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    codes = rng.integers(1, 11, size=(20, 30), dtype=np.uint8)
    return write_geotiff(
        tmp_path / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        codes,
        origin_e_m=400_000.0,
        origin_n_m=120_000.0,
    )


class TestScaleConfigShape:
    def test_a_square_configuration_is_unchanged(self) -> None:
        scale = parse_scale_config("50:1.0:10")
        assert scale.label == "50x50_f1.0_cap10"
        assert scale.shape == (50, 50)
        assert scale.n_cells == 2500

    def test_a_rectangular_configuration_parses(self) -> None:
        scale = parse_scale_config("210x240:1.0:10")
        assert scale.shape == (210, 240)
        assert scale.label == "210x240_f1.0_cap10"
        assert scale.n_cells == 210 * 240

    def test_the_label_reads_rows_by_columns(self) -> None:
        scale = ScaleConfig(
            grid_size=210, conversion_fraction=1.0, max_conversions=10, grid_width=240
        )
        assert scale.label.startswith("210x240")
        assert "210x240 grid" in scale.description

    def test_the_record_carries_both_dimensions(self) -> None:
        record = parse_scale_config("210x240:1.0:1").as_record()
        assert record["grid_height"] == 210
        assert record["grid_width"] == 240

    def test_a_malformed_grid_token_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="grid"):
            parse_scale_config("210x:1.0:10")


class TestEnvShape:
    def test_a_rectangular_grid_is_accepted_when_configured(self) -> None:
        grid = ParcelGrid.empty(4, 6)
        config = EnvConfig(grid_size=4, grid_width=6, iterations=1)
        env = UrbanLandUseEnv(grid, config=config)
        assert env.grid.shape == (4, 6)

    def test_a_mismatched_grid_is_still_rejected(self) -> None:
        grid = ParcelGrid.empty(4, 6)
        with pytest.raises(ValueError, match="grid_size"):
            UrbanLandUseEnv(grid, config=EnvConfig(grid_size=4, iterations=1))

    def test_the_square_default_is_unchanged(self) -> None:
        grid = ParcelGrid.empty(5, 5)
        env = UrbanLandUseEnv(grid, config=EnvConfig(grid_size=5, iterations=1))
        assert env.grid.shape == (5, 5)


class TestLoaderShape:
    def test_a_rectangular_window_is_cut(self, wide_class_raster: Path) -> None:
        raster = load_lcm_raster(wide_class_raster)
        grid = select_region(raster, shape=(8, 12), origin=(0, 0))
        assert grid.shape == (8, 12)

    def test_the_square_helper_still_works(self, wide_class_raster: Path) -> None:
        raster = load_lcm_raster(wide_class_raster)
        grid = select_region_of_interest(raster, grid_size=8, origin=(1, 2))
        assert grid.shape == (8, 8)

    def test_a_window_that_leaves_the_raster_is_rejected(
        self, wide_class_raster: Path
    ) -> None:
        raster = load_lcm_raster(wide_class_raster)
        with pytest.raises(ValueError, match="leaves the raster extent"):
            resolve_window_origin(raster, (25, 12), origin=(0, 0))

    def test_the_densest_window_respects_the_shape(
        self, wide_class_raster: Path
    ) -> None:
        raster = load_lcm_raster(wide_class_raster)
        row, col = resolve_window_origin(raster, (8, 12))
        assert 0 <= row <= 20 - 8
        assert 0 <= col <= 30 - 12


class TestRegionWindow:
    def test_the_documented_box_is_the_south_east_region(self) -> None:
        assert SOUTH_EAST_REGION.east_min == 410
        assert SOUTH_EAST_REGION.east_max == 650
        assert SOUTH_EAST_REGION.north_min == 60
        assert SOUTH_EAST_REGION.north_max == 270

    def test_the_region_window_is_two_hundred_forty_by_two_hundred_ten(self) -> None:
        window = region_window()
        assert window.width_km == 240
        assert window.rows == 210
        assert window.shape == (210, 240)
        assert window.label == SOUTH_EAST_REGION_LABEL

    def test_the_region_window_is_pinned_at_its_north_west_corner(self) -> None:
        assert region_window().bng_top_left_km == (410, 270)

    def test_a_square_scan_window_keeps_its_label(self) -> None:
        window = ScanWindow(easting_km=495, northing_km=140)
        assert window.label == "E495_N140"
        assert window.shape == (50, 50)

    def test_a_rectangular_window_without_a_name_says_its_shape(self) -> None:
        window = ScanWindow(easting_km=410, northing_km=60, size_km=240, height_km=210)
        assert window.label == "E410_N60_240x210"

    def test_the_region_label_round_trips_through_the_parser(self) -> None:
        assert parse_window_label(SOUTH_EAST_REGION_LABEL).label == (
            SOUTH_EAST_REGION_LABEL
        )
        assert parse_window_label(SOUTH_EAST_REGION_LABEL).width_km == 240


class TestRegionMask:
    def test_without_a_boundary_the_bounding_box_is_used(self, tmp_path: Path) -> None:
        from chapter4_qolu.data.region_mask import load_region_mask

        assert load_region_mask(None, shape=(4, 5), transform=None) is None

    def test_a_mask_is_applied_by_blanking_the_cover(self) -> None:
        from chapter4_qolu.data.region_mask import apply_region_mask

        grid = ParcelGrid.from_fractions(np.full((10, 2, 2), 0.1, dtype=np.float32))
        mask = np.array([[True, False], [True, True]])
        masked = apply_region_mask(grid, mask)
        assert masked.land_mask().tolist() == [[True, False], [True, True]]
        assert float(masked.composition[:, 0, 1].sum()) == 0.0

    def test_a_mask_of_the_wrong_shape_is_rejected(self) -> None:
        from chapter4_qolu.data.region_mask import apply_region_mask

        grid = ParcelGrid.empty(2, 2)
        with pytest.raises(ValueError, match="shape"):
            apply_region_mask(grid, np.ones((3, 3), dtype=bool))


class TestRunSpecRegion:
    def test_a_rectangular_run_spec_reports_its_shape(self, tmp_path: Path) -> None:
        from chapter4_qolu.scenario_worker import RunSpec

        spec = RunSpec(
            raster=tmp_path / "raster.img",
            method="mitm",
            scenario=1,
            seed=0,
            scale=parse_scale_config("210x240:1.0:10"),
            output_path=tmp_path / "out.json",
        )
        assert spec.grid_shape == (210, 240)
        assert spec.env_config().grid_width == 240

    def test_a_rectangular_run_executes(
        self, wide_class_raster: Path, tmp_path: Path
    ) -> None:
        from chapter4_qolu.scenario_worker import RunSpec, execute_run

        spec = RunSpec(
            raster=wide_class_raster,
            method="mitm",
            scenario=1,
            seed=0,
            scale=parse_scale_config("8x12:1.0:10"),
            output_path=tmp_path / "mitm_seed0.json",
            iterations=3,
            stakeholders=1,
            window_origin=(0, 0),
        )
        record = execute_run(spec)
        assert record["configuration"]["grid_width"] == 12
        assert record["env_settings"]["grid_width"] == 12
        from chapter4_qolu.run_record import decode_class_grid

        assert decode_class_grid(record["final_grid"]).shape == (8, 12)


class TestCampaignRegionOption:
    def test_the_region_flag_pins_the_whole_box(self, tmp_path: Path) -> None:
        from chapter4_qolu.run_scenarios import _build_parser, _settings_from

        arguments = _build_parser().parse_args(
            [
                "--region",
                "--configs",
                "210x240:1.0:10",
                "--output-dir",
                str(tmp_path),
            ]
        )
        settings = _settings_from(arguments, tmp_path / "raster.img")
        assert settings.window_label == SOUTH_EAST_REGION_LABEL
        assert settings.window_bng_km == (410.0, 270.0)

    def test_the_cost_model_scales_with_cells_not_with_a_side(self) -> None:
        from chapter4_qolu.campaign_plan import _grid_cost_factor

        assert _grid_cost_factor(2500) == pytest.approx(1.0)
        assert _grid_cost_factor(10_000) == pytest.approx(1.25)
