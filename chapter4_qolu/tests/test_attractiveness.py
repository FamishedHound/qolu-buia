"""Tests for the attractiveness metric (Section 3.2 / Eq. 1 of the paper)."""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.env.attractiveness import AttractivenessConfig, AttractivenessField
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.land_cover import (
    N_AGGREGATE_CLASSES,
    AggregateClass,
    LandCoverGroup,
    band_index,
)


def blank(height: int = 7, width: int = 7) -> np.ndarray:
    return np.zeros((N_AGGREGATE_CLASSES, height, width), dtype=np.float32)


def grid_with(
    features: dict[tuple[int, int], AggregateClass], size: int = 7
) -> ParcelGrid:
    composition = blank(size, size)
    composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 1.0
    for (row, col), land_class in features.items():
        composition[:, row, col] = 0.0
        composition[band_index(land_class), row, col] = 1.0
    return ParcelGrid(composition)


class TestAttractivenessConfig:
    def test_default_weights_cover_the_paper_features(self) -> None:
        config = AttractivenessConfig()
        groups = {group for group, _ in config.group_weights}
        assert groups == {
            LandCoverGroup.WOODLAND,
            LandCoverGroup.FRESHWATER,
            LandCoverGroup.URBAN,
        }
        assert sum(weight for _, weight in config.group_weights) == pytest.approx(1.0)

    def test_rejects_non_positive_radius_or_decay(self) -> None:
        with pytest.raises(ValueError):
            AttractivenessConfig(radius=0)
        with pytest.raises(ValueError):
            AttractivenessConfig(decay_length=0.0)


class TestAttractivenessScores:
    def test_scores_are_bounded_in_the_unit_interval(self) -> None:
        grid = grid_with(
            {
                (0, 0): AggregateClass.BUILT_UP,
                (3, 3): AggregateClass.FRESHWATER,
                (6, 6): AggregateClass.BROADLEAF_WOODLAND,
            }
        )
        field = AttractivenessField.from_grid(grid, AttractivenessConfig())
        assert field.scores.shape == (7, 7)
        assert float(field.scores.min()) >= 0.0
        assert float(field.scores.max()) <= 1.0

    def test_parcels_near_desirable_features_score_higher(self) -> None:
        grid = grid_with(
            {
                (1, 1): AggregateClass.BUILT_UP,
                (1, 2): AggregateClass.FRESHWATER,
                (2, 1): AggregateClass.BROADLEAF_WOODLAND,
            }
        )
        field = AttractivenessField.from_grid(grid, AttractivenessConfig(radius=2))
        near = field.scores[2, 2]
        far = field.scores[6, 6]
        assert near > far

    def test_all_three_feature_groups_contribute(self) -> None:
        config = AttractivenessConfig(radius=1, decay_length=1.0)
        baseline = AttractivenessField.from_grid(grid_with({}), config).scores[3, 3]
        for land_class in (
            AggregateClass.BROADLEAF_WOODLAND,
            AggregateClass.FRESHWATER,
            AggregateClass.BUILT_UP,
        ):
            grid = grid_with({(3, 4): land_class})
            score = AttractivenessField.from_grid(grid, config).scores[3, 3]
            assert score > baseline

    def test_closer_features_contribute_more_than_distant_ones(self) -> None:
        config = AttractivenessConfig(radius=3, decay_length=1.0)
        close = AttractivenessField.from_grid(
            grid_with({(3, 4): AggregateClass.BUILT_UP}), config
        ).scores[3, 3]
        distant = AttractivenessField.from_grid(
            grid_with({(3, 6): AggregateClass.BUILT_UP}), config
        ).scores[3, 3]
        assert close > distant

    def test_feature_free_landscape_has_zero_attractiveness(self) -> None:
        field = AttractivenessField.from_grid(grid_with({}), AttractivenessConfig())
        np.testing.assert_allclose(field.scores, 0.0, atol=1e-7)


class TestLocalUpdate:
    def test_local_update_matches_a_full_recompute(self) -> None:
        config = AttractivenessConfig(radius=2, decay_length=1.5)
        grid = grid_with(
            {
                (0, 0): AggregateClass.BUILT_UP,
                (2, 5): AggregateClass.FRESHWATER,
                (5, 2): AggregateClass.CONIFEROUS_WOODLAND,
            }
        )
        field = AttractivenessField.from_grid(grid, config)
        changed = [ParcelIndex(3, 3), ParcelIndex(4, 1)]
        new_grid, _ = grid.convert_to_urban(changed, fraction=1.0)

        incremental = field.updated(new_grid, changed)
        full = AttractivenessField.from_grid(new_grid, config)
        np.testing.assert_allclose(incremental.scores, full.scores, atol=1e-6)

    def test_local_update_matches_a_full_recompute_at_default_settings(self) -> None:
        # Same check on the configuration the environment actually runs with
        # (radius 5, decay 2 km), including parcels changed next to the border.
        config = AttractivenessConfig()
        rng = np.random.default_rng(7)
        composition = np.zeros((N_AGGREGATE_CLASSES, 20, 20), dtype=np.float32)
        for row in range(20):
            for col in range(20):
                composition[rng.integers(0, N_AGGREGATE_CLASSES), row, col] = 1.0
        grid = ParcelGrid(composition)
        field = AttractivenessField.from_grid(grid, config)

        changed = [ParcelIndex(0, 0), ParcelIndex(9, 11), ParcelIndex(19, 18)]
        new_grid, conversions = grid.convert_to_urban(changed, fraction=1.0)
        assert conversions

        incremental = field.updated(new_grid, [c.parcel for c in conversions])
        full = AttractivenessField.from_grid(new_grid, config)
        np.testing.assert_allclose(incremental.scores, full.scores, atol=1e-12)

    def test_update_is_pure(self) -> None:
        config = AttractivenessConfig(radius=1)
        grid = grid_with({(0, 0): AggregateClass.BUILT_UP})
        field = AttractivenessField.from_grid(grid, config)
        before = field.scores.copy()
        new_grid, _ = grid.convert_to_urban([ParcelIndex(3, 3)], fraction=1.0)
        updated = field.updated(new_grid, [ParcelIndex(3, 3)])
        assert updated is not field
        np.testing.assert_allclose(field.scores, before)

    def test_only_parcels_within_the_radius_change(self) -> None:
        config = AttractivenessConfig(radius=1, decay_length=1.0)
        grid = grid_with({(0, 0): AggregateClass.BUILT_UP})
        field = AttractivenessField.from_grid(grid, config)
        new_grid, _ = grid.convert_to_urban([ParcelIndex(3, 3)], fraction=1.0)
        updated = field.updated(new_grid, [ParcelIndex(3, 3)])
        difference = np.abs(updated.scores - field.scores)
        assert difference[3, 3] > 0.0
        assert difference[6, 6] == pytest.approx(0.0)

    def test_conversion_raises_neighbour_attractiveness(self) -> None:
        config = AttractivenessConfig(radius=2, decay_length=1.5)
        grid = grid_with({})
        field = AttractivenessField.from_grid(grid, config)
        new_grid, _ = grid.convert_to_urban([ParcelIndex(3, 3)], fraction=1.0)
        updated = field.updated(new_grid, [ParcelIndex(3, 3)])
        assert updated.scores[3, 4] > field.scores[3, 4]


class TestEligibility:
    def test_urban_and_nodata_parcels_are_not_development_candidates(self) -> None:
        composition = blank(3, 3)
        composition[band_index(AggregateClass.ARABLE), 0, 0] = 1.0
        composition[band_index(AggregateClass.BUILT_UP), 1, 1] = 1.0
        grid = ParcelGrid(composition)
        eligible = grid.convertible_mask(urban_saturation=0.99)
        assert eligible[0, 0]
        assert not eligible[1, 1]  # already urban
        assert not eligible[2, 2]  # no data
