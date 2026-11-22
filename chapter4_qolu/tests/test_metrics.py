"""Tests for the conversion-alignment metric of the paper (C_align)."""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.env.metrics import commonality, conversion_alignment
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index


def uniform_grid(
    land_class: AggregateClass = AggregateClass.ARABLE, size: int = 3
) -> ParcelGrid:
    composition = np.zeros((N_AGGREGATE_CLASSES, size, size), dtype=np.float32)
    composition[band_index(land_class)] = 1.0
    return ParcelGrid(composition)


def mixed_grid() -> ParcelGrid:
    """3x3 grid: five arable parcels, three woodland, centre semi-natural."""
    layout = [
        [AggregateClass.ARABLE, AggregateClass.ARABLE, AggregateClass.ARABLE],
        [
            AggregateClass.BROADLEAF_WOODLAND,
            AggregateClass.SEMI_NATURAL_GRASSLAND,
            AggregateClass.ARABLE,
        ],
        [
            AggregateClass.BROADLEAF_WOODLAND,
            AggregateClass.BROADLEAF_WOODLAND,
            AggregateClass.ARABLE,
        ],
    ]
    composition = np.zeros((N_AGGREGATE_CLASSES, 3, 3), dtype=np.float32)
    for row, classes in enumerate(layout):
        for col, land_class in enumerate(classes):
            composition[band_index(land_class), row, col] = 1.0
    return ParcelGrid(composition)


class TestCommonality:
    def test_frequency_of_the_most_common_neighbouring_type(self) -> None:
        classes = mixed_grid().dominant_class()
        # Centre has 8 neighbours: 5 arable, 3 woodland.
        assert commonality(classes, ParcelIndex(1, 1), radius=1) == pytest.approx(5 / 8)

    def test_uniform_neighbourhood_scores_one(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 3, 3), dtype=np.float32)
        composition[band_index(AggregateClass.ARABLE)] = 1.0
        classes = ParcelGrid(composition).dominant_class()
        assert commonality(classes, ParcelIndex(1, 1), radius=1) == pytest.approx(1.0)

    def test_corner_parcels_only_count_in_bounds_neighbours(self) -> None:
        classes = mixed_grid().dominant_class()
        # Corner (0, 0) has 3 neighbours: arable (0,1), woodland (1,0), semi-natural (1,1).
        assert commonality(classes, ParcelIndex(0, 0), radius=1) == pytest.approx(1 / 3)

    def test_excludes_the_focal_parcel(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 1, 2), dtype=np.float32)
        composition[band_index(AggregateClass.ARABLE), 0, 0] = 1.0
        composition[band_index(AggregateClass.BUILT_UP), 0, 1] = 1.0
        classes = ParcelGrid(composition).dominant_class()
        assert commonality(classes, ParcelIndex(0, 0), radius=1) == pytest.approx(1.0)

    def test_no_data_neighbours_are_ignored(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 1, 3), dtype=np.float32)
        composition[band_index(AggregateClass.ARABLE), 0, 0] = 1.0
        composition[band_index(AggregateClass.ARABLE), 0, 1] = 1.0
        classes = ParcelGrid(composition).dominant_class()
        # Parcel (0, 1) has one arable neighbour and one no-data neighbour; the
        # no-data parcel must not be treated as a land type of its own (which
        # would give 1/2 instead of 1/1).
        assert commonality(classes, ParcelIndex(0, 1), radius=1) == pytest.approx(1.0)

    def test_isolated_parcel_has_zero_commonality(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 1, 2), dtype=np.float32)
        composition[band_index(AggregateClass.ARABLE), 0, 0] = 1.0
        classes = ParcelGrid(composition).dominant_class()
        assert commonality(classes, ParcelIndex(0, 0), radius=1) == pytest.approx(0.0)


class TestConversionAlignment:
    def test_identical_states_align_perfectly(self) -> None:
        grid = mixed_grid()
        assert conversion_alignment(grid, grid) == pytest.approx(0.0)

    def test_changes_increase_the_misalignment(self) -> None:
        grid = uniform_grid()
        converted, _ = grid.convert_to_urban([ParcelIndex(1, 1)], fraction=1.0)
        assert conversion_alignment(grid, converted) > 0.0

    def test_is_blind_to_changes_that_do_not_move_the_modal_neighbour(self) -> None:
        # Published behaviour worth pinning down: C_align compares the frequency
        # of the *most common* neighbouring land type, so swapping one singleton
        # land type for another leaves it unchanged.
        grid = mixed_grid()
        converted, _ = grid.convert_to_urban([ParcelIndex(1, 1)], fraction=1.0)
        assert conversion_alignment(grid, converted) == pytest.approx(0.0)

    def test_matches_the_mean_squared_commonality_difference(self) -> None:
        grid = uniform_grid()
        converted, _ = grid.convert_to_urban([ParcelIndex(1, 1)], fraction=1.0)
        before_classes = grid.dominant_class()
        after_classes = converted.dominant_class()
        expected = float(
            np.mean(
                [
                    (
                        commonality(before_classes, ParcelIndex(row, col))
                        - commonality(after_classes, ParcelIndex(row, col))
                    )
                    ** 2
                    for row in range(3)
                    for col in range(3)
                ]
            )
        )
        assert conversion_alignment(grid, converted) == pytest.approx(expected)

    def test_rejects_shape_mismatch(self) -> None:
        small = ParcelGrid(np.zeros((N_AGGREGATE_CLASSES, 2, 2), dtype=np.float32))
        with pytest.raises(ValueError):
            conversion_alignment(mixed_grid(), small)
