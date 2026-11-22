"""Tests for the well-being metric WB = H_occupied / H_total (Section 3.6)."""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.config import AttractivenessConfig, WellBeingConfig
from chapter4_qolu.env.attractiveness import AttractivenessField
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.env.well_being import (
    housing_capacity,
    occupied_housing,
    population_at,
    well_being,
)
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index


def urban_grid(urban_fractions: list[list[float]]) -> ParcelGrid:
    array = np.asarray(urban_fractions, dtype=np.float32)
    composition = np.zeros((N_AGGREGATE_CLASSES,) + array.shape, dtype=np.float32)
    composition[band_index(AggregateClass.BUILT_UP)] = array
    composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 1.0 - array
    return ParcelGrid(composition)


def flat_field(grid: ParcelGrid, value: float = 0.5) -> AttractivenessField:
    return AttractivenessField(
        scores=np.full(grid.shape, value, dtype=np.float64),
        config=AttractivenessConfig(),
    )


class TestHousingCapacity:
    def test_capacity_is_proportional_to_urban_cover(self) -> None:
        grid = urban_grid([[1.0, 0.5], [0.0, 0.25]])
        config = WellBeingConfig(capacity_per_parcel=1000.0)
        capacity = housing_capacity(grid, config)
        np.testing.assert_allclose(capacity, [[1000.0, 500.0], [0.0, 250.0]])

    def test_total_capacity_is_the_sum(self) -> None:
        grid = urban_grid([[1.0, 1.0]])
        config = WellBeingConfig(capacity_per_parcel=10.0)
        assert housing_capacity(grid, config).sum() == pytest.approx(20.0)


class TestOccupancy:
    def test_demand_follows_attractiveness_and_is_capped_by_capacity(self) -> None:
        grid = urban_grid([[1.0, 1.0]])
        scores = np.array([[3.0, 1.0]])
        field = AttractivenessField(scores=scores, config=AttractivenessConfig())
        config = WellBeingConfig(capacity_per_parcel=100.0)
        occupied = occupied_housing(grid, field, population=100.0, config=config)
        # 75 / 25 split by attractiveness, both below the 100-home capacity.
        np.testing.assert_allclose(occupied, [[75.0, 25.0]])

    def test_overflow_demand_is_lost_to_the_capacity_limit(self) -> None:
        grid = urban_grid([[1.0, 1.0]])
        scores = np.array([[9.0, 1.0]])
        field = AttractivenessField(scores=scores, config=AttractivenessConfig())
        config = WellBeingConfig(capacity_per_parcel=100.0)
        occupied = occupied_housing(grid, field, population=200.0, config=config)
        # 180 households want the attractive parcel but only 100 homes exist.
        np.testing.assert_allclose(occupied, [[100.0, 20.0]])

    def test_no_urban_land_means_no_occupancy(self) -> None:
        grid = urban_grid([[0.0, 0.0]])
        config = WellBeingConfig()
        occupied = occupied_housing(grid, flat_field(grid), 100.0, config)
        np.testing.assert_allclose(occupied, 0.0)


class TestWellBeing:
    def test_full_occupancy_scores_one(self) -> None:
        grid = urban_grid([[1.0, 1.0]])
        config = WellBeingConfig(capacity_per_parcel=50.0)
        assert well_being(grid, flat_field(grid), 100.0, config) == pytest.approx(1.0)

    def test_half_occupancy_scores_one_half(self) -> None:
        grid = urban_grid([[1.0, 1.0]])
        config = WellBeingConfig(capacity_per_parcel=50.0)
        assert well_being(grid, flat_field(grid), 50.0, config) == pytest.approx(0.5)

    def test_spatial_mismatch_lowers_well_being(self) -> None:
        grid = urban_grid([[1.0, 1.0]])
        config = WellBeingConfig(capacity_per_parcel=50.0)
        skewed = AttractivenessField(
            scores=np.array([[9.0, 1.0]]), config=AttractivenessConfig()
        )
        balanced = flat_field(grid)
        assert well_being(grid, skewed, 100.0, config) < well_being(
            grid, balanced, 100.0, config
        )

    def test_is_bounded_in_the_unit_interval(self) -> None:
        grid = urban_grid([[1.0, 0.5], [0.25, 0.0]])
        config = WellBeingConfig(capacity_per_parcel=10.0)
        for population in (0.0, 5.0, 50.0, 5000.0):
            score = well_being(grid, flat_field(grid), population, config)
            assert 0.0 <= score <= 1.0

    def test_landscape_without_housing_scores_zero(self) -> None:
        grid = urban_grid([[0.0, 0.0]])
        assert well_being(grid, flat_field(grid), 10.0, WellBeingConfig()) == (
            pytest.approx(0.0)
        )

    def test_rejects_negative_population(self) -> None:
        grid = urban_grid([[1.0]])
        with pytest.raises(ValueError):
            well_being(grid, flat_field(grid), -1.0, WellBeingConfig())


class TestPopulationDynamics:
    def test_initial_population_is_a_share_of_the_starting_capacity(self) -> None:
        config = WellBeingConfig(initial_occupancy=0.9)
        assert population_at(0, initial_capacity=1000.0, config=config) == (
            pytest.approx(900.0)
        )

    def test_population_grows_with_the_iteration_count(self) -> None:
        config = WellBeingConfig(
            initial_occupancy=0.9, population_growth_per_iteration=0.01
        )
        assert population_at(10, initial_capacity=1000.0, config=config) == (
            pytest.approx(900.0 * 1.1)
        )

    def test_static_population_by_default(self) -> None:
        config = WellBeingConfig()
        assert config.population_growth_per_iteration == pytest.approx(0.0)
        assert population_at(500, 1000.0, config) == population_at(0, 1000.0, config)
