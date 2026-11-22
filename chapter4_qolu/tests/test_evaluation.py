"""Tests for the published evaluation metrics (Section 4, Tables 4.4-4.7)."""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.config import AttractivenessConfig, RewardConfig, WellBeingConfig
from chapter4_qolu.env.attractiveness import AttractivenessField
from chapter4_qolu.env.evaluation import (
    distance_to_desired_land_types,
    evaluate_landscape,
    newly_developed_mask,
    preservation_of_land_types,
    urban_connectivity,
)
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index


def grid_with(
    features: dict[tuple[int, int], AggregateClass], size: int = 5
) -> ParcelGrid:
    composition = np.zeros((N_AGGREGATE_CLASSES, size, size), dtype=np.float32)
    composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 1.0
    for (row, col), land_class in features.items():
        composition[:, row, col] = 0.0
        composition[band_index(land_class), row, col] = 1.0
    return ParcelGrid(composition)


class TestNewlyDevelopedMask:
    def test_flags_parcels_that_became_urban(self) -> None:
        initial = grid_with({(0, 0): AggregateClass.BUILT_UP})
        final, _ = initial.convert_to_urban([ParcelIndex(2, 2)], fraction=1.0)
        mask = newly_developed_mask(initial, final, threshold=0.5)
        assert mask[2, 2]
        assert not mask[0, 0]  # already urban before the run
        assert mask.sum() == 1

    def test_no_change_gives_an_empty_mask(self) -> None:
        grid = grid_with({})
        assert not newly_developed_mask(grid, grid, threshold=0.5).any()

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            newly_developed_mask(grid_with({}), grid_with({}, size=4), threshold=0.5)


class TestDistanceToDesiredLandTypes:
    def test_development_far_from_habitat_scores_higher(self) -> None:
        initial = grid_with({(0, 0): AggregateClass.MOUNTAIN_HEATH_BOG})
        near, _ = initial.convert_to_urban([ParcelIndex(0, 1)], fraction=1.0)
        far, _ = initial.convert_to_urban([ParcelIndex(4, 4)], fraction=1.0)
        config = RewardConfig()
        assert distance_to_desired_land_types(
            initial, far, config
        ) > distance_to_desired_land_types(initial, near, config)

    def test_is_normalised_to_the_unit_interval(self) -> None:
        initial = grid_with({(0, 0): AggregateClass.MOUNTAIN_HEATH_BOG})
        final, _ = initial.convert_to_urban([ParcelIndex(3, 3)], fraction=1.0)
        score = distance_to_desired_land_types(initial, final, RewardConfig())
        assert 0.0 <= score <= 1.0

    def test_building_on_top_of_the_habitat_scores_zero(self) -> None:
        initial = grid_with({(2, 2): AggregateClass.MOUNTAIN_HEATH_BOG})
        final, _ = initial.convert_to_urban([ParcelIndex(2, 2)], fraction=1.0)
        assert distance_to_desired_land_types(initial, final, RewardConfig()) == (
            pytest.approx(0.0)
        )

    def test_no_development_is_reported_as_no_harm(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.MOUNTAIN_HEATH_BOG})
        assert distance_to_desired_land_types(grid, grid, RewardConfig()) == (
            pytest.approx(1.0)
        )


class TestPreservationOfLandTypes:
    def test_untouched_landscape_scores_one(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.BROADLEAF_WOODLAND})
        assert preservation_of_land_types(grid, grid, RewardConfig()) == (
            pytest.approx(1.0)
        )

    def test_losing_half_the_protected_area_scores_one_half(self) -> None:
        initial = grid_with(
            {
                (0, 0): AggregateClass.BROADLEAF_WOODLAND,
                (0, 1): AggregateClass.ARABLE,
            }
        )
        final, _ = initial.convert_to_urban([ParcelIndex(0, 0)], fraction=1.0)
        assert preservation_of_land_types(initial, final, RewardConfig()) == (
            pytest.approx(0.5)
        )

    def test_absent_protected_land_counts_as_preserved(self) -> None:
        grid = grid_with({})
        final, _ = grid.convert_to_urban([ParcelIndex(1, 1)], fraction=1.0)
        assert preservation_of_land_types(grid, final, RewardConfig()) == (
            pytest.approx(1.0)
        )


class TestUrbanConnectivity:
    def test_development_next_to_town_scores_higher(self) -> None:
        initial = grid_with({(0, 0): AggregateClass.BUILT_UP})
        near, _ = initial.convert_to_urban([ParcelIndex(0, 1)], fraction=1.0)
        far, _ = initial.convert_to_urban([ParcelIndex(4, 4)], fraction=1.0)
        config = RewardConfig()
        assert urban_connectivity(initial, near, config) > urban_connectivity(
            initial, far, config
        )

    def test_is_normalised_to_the_unit_interval(self) -> None:
        initial = grid_with({(0, 0): AggregateClass.BUILT_UP})
        final, _ = initial.convert_to_urban([ParcelIndex(2, 3)], fraction=1.0)
        assert 0.0 <= urban_connectivity(initial, final, RewardConfig()) <= 1.0

    def test_development_without_any_existing_town_scores_zero(self) -> None:
        initial = grid_with({})
        final, _ = initial.convert_to_urban([ParcelIndex(2, 2)], fraction=1.0)
        assert urban_connectivity(initial, final, RewardConfig()) == pytest.approx(0.0)

    def test_no_development_is_reported_as_no_harm(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.BUILT_UP})
        assert urban_connectivity(grid, grid, RewardConfig()) == pytest.approx(1.0)


class TestEvaluateLandscape:
    def test_reports_every_published_metric(self) -> None:
        initial = grid_with(
            {
                (0, 0): AggregateClass.BUILT_UP,
                (4, 4): AggregateClass.MOUNTAIN_HEATH_BOG,
                (0, 4): AggregateClass.BROADLEAF_WOODLAND,
            }
        )
        final, _ = initial.convert_to_urban([ParcelIndex(1, 1)], fraction=1.0)
        field = AttractivenessField.from_grid(final, AttractivenessConfig())
        metrics = evaluate_landscape(
            initial,
            final,
            attractiveness=field,
            population=10.0,
            reward_config=RewardConfig(),
            well_being_config=WellBeingConfig(),
        )
        assert set(metrics) == {
            "land_use",
            "woodland",
            "urban",
            "conversion_alignment",
            "c_align",
            "well_being",
        }
        assert all(np.isfinite(value) for value in metrics.values())
        for key in ("land_use", "woodland", "urban", "c_align", "well_being"):
            assert 0.0 <= metrics[key] <= 1.0

    def test_reports_the_corrected_conversion_metric(self) -> None:
        from chapter4_qolu.env.metrics import c_align

        initial = grid_with({(1, 1): AggregateClass.BROADLEAF_WOODLAND})
        final, _ = initial.convert_to_urban([ParcelIndex(1, 1)], fraction=1.0)
        field = AttractivenessField.from_grid(final, AttractivenessConfig())
        metrics = evaluate_landscape(
            initial,
            final,
            attractiveness=field,
            population=10.0,
            reward_config=RewardConfig(),
            well_being_config=WellBeingConfig(),
        )
        assert metrics["c_align"] == c_align(initial, final)
        assert metrics["c_align"] > 0.0

    def test_metric_names_match_the_result_table_columns(self) -> None:
        initial = grid_with({})
        field = AttractivenessField.from_grid(initial, AttractivenessConfig())
        metrics = evaluate_landscape(
            initial,
            initial,
            attractiveness=field,
            population=0.0,
            reward_config=RewardConfig(),
            well_being_config=WellBeingConfig(),
        )
        # Tables 4.4-4.7 label the three stakeholder columns exactly this way.
        assert {"land_use", "woodland", "urban"} <= set(metrics)
