"""Tests for the parcel grid state container."""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.land_cover import (
    N_AGGREGATE_CLASSES,
    AggregateClass,
    LandCoverGroup,
    band_index,
    group_indices,
)


def make_grid(height: int = 3, width: int = 3) -> ParcelGrid:
    composition = np.zeros((N_AGGREGATE_CLASSES, height, width), dtype=np.float32)
    composition[band_index(AggregateClass.BROADLEAF_WOODLAND)] = 0.5
    composition[band_index(AggregateClass.ARABLE)] = 0.5
    return ParcelGrid(composition)


class TestLandCoverTaxonomy:
    def test_ten_aggregate_classes(self) -> None:
        assert N_AGGREGATE_CLASSES == 10
        assert len(list(AggregateClass)) == 10

    def test_class_codes_follow_the_ukceh_table(self) -> None:
        assert AggregateClass.BROADLEAF_WOODLAND == 1
        assert AggregateClass.CONIFEROUS_WOODLAND == 2
        assert AggregateClass.ARABLE == 3
        assert AggregateClass.IMPROVED_GRASSLAND == 4
        assert AggregateClass.SEMI_NATURAL_GRASSLAND == 5
        assert AggregateClass.MOUNTAIN_HEATH_BOG == 6
        assert AggregateClass.SALTWATER == 7
        assert AggregateClass.FRESHWATER == 8
        assert AggregateClass.COASTAL == 9
        assert AggregateClass.BUILT_UP == 10

    def test_band_index_is_zero_based(self) -> None:
        assert band_index(AggregateClass.BROADLEAF_WOODLAND) == 0
        assert band_index(AggregateClass.BUILT_UP) == 9

    def test_group_membership(self) -> None:
        assert group_indices(LandCoverGroup.WOODLAND) == (0, 1)
        assert group_indices(LandCoverGroup.ARABLE) == (2,)
        assert group_indices(LandCoverGroup.AGRICULTURAL) == (2, 3)
        assert group_indices(LandCoverGroup.FRESHWATER) == (7,)
        assert group_indices(LandCoverGroup.URBAN) == (9,)
        assert group_indices(LandCoverGroup.MOUNTAIN_HEATH_BOG) == (5,)


class TestParcelGridBasics:
    def test_shape_properties(self) -> None:
        grid = make_grid(4, 6)
        assert grid.height == 4
        assert grid.width == 6
        assert grid.n_classes == N_AGGREGATE_CLASSES
        assert grid.n_parcels == 24

    def test_rejects_wrong_rank_or_class_count(self) -> None:
        with pytest.raises(ValueError):
            ParcelGrid(np.zeros((3, 3), dtype=np.float32))
        with pytest.raises(ValueError):
            ParcelGrid(np.zeros((5, 3, 3), dtype=np.float32))

    def test_rejects_out_of_range_fractions(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 2, 2), dtype=np.float32)
        composition[0, 0, 0] = 1.5
        with pytest.raises(ValueError):
            ParcelGrid(composition)

    def test_group_fraction_sums_member_classes(self) -> None:
        grid = make_grid()
        woodland = grid.group_fraction(LandCoverGroup.WOODLAND)
        assert woodland.shape == (3, 3)
        np.testing.assert_allclose(woodland, 0.5)

    def test_group_mask_uses_a_presence_threshold(self) -> None:
        grid = make_grid()
        assert grid.group_mask(LandCoverGroup.WOODLAND, threshold=0.4).all()
        assert not grid.group_mask(LandCoverGroup.WOODLAND, threshold=0.6).any()

    def test_dominant_class_marks_empty_parcels_as_nodata(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 2, 2), dtype=np.float32)
        composition[band_index(AggregateClass.FRESHWATER), 0, 0] = 0.9
        grid = ParcelGrid(composition)
        dominant = grid.dominant_class()
        assert dominant[0, 0] == band_index(AggregateClass.FRESHWATER)
        assert dominant[1, 1] == ParcelGrid.NO_DATA

    def test_land_mask_excludes_empty_parcels(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 2, 2), dtype=np.float32)
        composition[0, 0, 0] = 1.0
        grid = ParcelGrid(composition)
        assert grid.land_mask().tolist() == [[True, False], [False, False]]


class TestConversion:
    def test_returns_a_new_grid_and_leaves_the_original_untouched(self) -> None:
        grid = make_grid()
        before = grid.composition.copy()
        new_grid, _ = grid.convert_to_urban([ParcelIndex(1, 1)], fraction=1.0)
        assert new_grid is not grid
        np.testing.assert_allclose(grid.composition, before)

    def test_full_conversion_moves_all_cover_to_built_up(self) -> None:
        grid = make_grid()
        new_grid, _ = grid.convert_to_urban([ParcelIndex(0, 0)], fraction=1.0)
        urban = band_index(AggregateClass.BUILT_UP)
        assert new_grid.composition[urban, 0, 0] == pytest.approx(1.0)
        assert new_grid.composition[
            band_index(AggregateClass.ARABLE), 0, 0
        ] == pytest.approx(0.0)

    def test_partial_conversion_scales_non_urban_cover_proportionally(self) -> None:
        grid = make_grid()
        new_grid, _ = grid.convert_to_urban([ParcelIndex(0, 0)], fraction=0.5)
        urban = band_index(AggregateClass.BUILT_UP)
        assert new_grid.composition[urban, 0, 0] == pytest.approx(0.5)
        assert new_grid.composition[
            band_index(AggregateClass.ARABLE), 0, 0
        ] == pytest.approx(0.25)
        assert new_grid.composition[:, 0, 0].sum() == pytest.approx(1.0)

    def test_conversion_records_the_cover_that_was_lost(self) -> None:
        grid = make_grid()
        _, conversions = grid.convert_to_urban([ParcelIndex(2, 2)], fraction=1.0)
        assert len(conversions) == 1
        conversion = conversions[0]
        assert conversion.parcel == ParcelIndex(2, 2)
        assert conversion.urban_gain == pytest.approx(1.0)
        assert conversion.lost_cover[
            band_index(AggregateClass.ARABLE)
        ] == pytest.approx(0.5)
        assert conversion.lost_cover[
            band_index(AggregateClass.BUILT_UP)
        ] == pytest.approx(0.0)

    def test_duplicate_parcels_are_converted_once(self) -> None:
        grid = make_grid()
        new_grid, conversions = grid.convert_to_urban(
            [ParcelIndex(0, 1), ParcelIndex(0, 1)], fraction=1.0
        )
        assert len(conversions) == 1
        urban = band_index(AggregateClass.BUILT_UP)
        assert new_grid.composition[urban, 0, 1] == pytest.approx(1.0)

    def test_already_urban_parcels_are_skipped(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 2, 2), dtype=np.float32)
        composition[band_index(AggregateClass.BUILT_UP), 0, 0] = 1.0
        grid = ParcelGrid(composition)
        _, conversions = grid.convert_to_urban([ParcelIndex(0, 0)], fraction=1.0)
        assert conversions == ()

    def test_empty_nodata_parcels_are_skipped(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 2, 2), dtype=np.float32)
        grid = ParcelGrid(composition)
        _, conversions = grid.convert_to_urban([ParcelIndex(1, 0)], fraction=1.0)
        assert conversions == ()

    def test_rejects_invalid_fraction(self) -> None:
        grid = make_grid()
        with pytest.raises(ValueError):
            grid.convert_to_urban([ParcelIndex(0, 0)], fraction=1.5)

    def test_rejects_out_of_bounds_parcels(self) -> None:
        grid = make_grid()
        with pytest.raises(IndexError):
            grid.convert_to_urban([ParcelIndex(9, 9)], fraction=1.0)


class TestObservationPatches:
    def test_patch_is_centred_and_zero_padded_at_the_border(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 3, 3), dtype=np.float32)
        composition[0] = np.arange(9, dtype=np.float32).reshape(3, 3) / 10.0
        grid = ParcelGrid(composition)
        patch = grid.patch(ParcelIndex(0, 0), window=3)
        assert patch.shape == (N_AGGREGATE_CLASSES, 3, 3)
        np.testing.assert_allclose(patch[0, 0], [0.0, 0.0, 0.0])
        assert patch[0, 1, 1] == pytest.approx(0.0)
        assert patch[0, 2, 2] == pytest.approx(0.4)

    def test_rejects_even_window(self) -> None:
        grid = make_grid()
        with pytest.raises(ValueError):
            grid.patch(ParcelIndex(1, 1), window=4)
