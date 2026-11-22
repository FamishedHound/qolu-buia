"""The total-variation conversion metric ``C_align``.

    C_align(s, s_hat) = 1/|P| sum_j 1/2 sum_c | p_jc - p_hat_jc |

``p_jc`` is the share of the parcels adjacent to parcel ``j`` (up to eight)
whose land type is ``c``: the mean total-variation distance between each
parcel's neighbourhood land-type distribution before and after.  The
neighbourhood is the one the published ``conversion_alignment`` already uses -
the eight adjacent parcels, the focal parcel excluded, fewer at the border, and
parcels without land cover left out of the counts.

The published definition compares only the frequency of the most common
neighbouring type, so a neighbourhood that goes from six-of-eight woodland to
six-of-eight urban scores 0, the best value; the tests pin that the
total-variation term is 0.75 and that the published one is 0.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from chapter4_qolu.env.metrics import (
    c_align,
    c_align_from_classes,
    c_align_map,
    commonality_map,
)
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index

W = band_index(AggregateClass.BROADLEAF_WOODLAND)
A = band_index(AggregateClass.ARABLE)
G = band_index(AggregateClass.IMPROVED_GRASSLAND)
U = band_index(AggregateClass.BUILT_UP)
NO_DATA = ParcelGrid.NO_DATA


def six_of_eight() -> tuple[np.ndarray, np.ndarray]:
    """Centre arable; six woodland and two grassland neighbours, then urban."""
    before = np.array([[W, W, W], [W, A, W], [G, W, G]], dtype=np.int64)
    after = np.where(before == W, U, before)
    return before, after


def reference_terms(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """A deliberately naive per-parcel implementation of the definition."""
    rows, cols = before.shape
    terms = np.zeros(before.shape, dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            neighbours = [
                (row + dr, col + dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
                and 0 <= row + dr < rows
                and 0 <= col + dc < cols
                and before[row + dr, col + dc] >= 0
            ]
            if not neighbours:
                continue
            first = Counter(int(before[r, c]) for r, c in neighbours)
            second = Counter(int(after[r, c]) for r, c in neighbours)
            n = len(neighbours)
            terms[row, col] = 0.5 * sum(
                abs(first[k] / n - second[k] / n) for k in set(first) | set(second)
            )
    return terms


def random_pair(
    rng: np.random.Generator, shape: tuple[int, int] = (7, 9), no_data: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    before = rng.integers(0, N_AGGREGATE_CLASSES, size=shape).astype(np.int64)
    before[rng.random(shape) < no_data] = NO_DATA
    converted = (rng.random(shape) < 0.3) & (before >= 0)
    after = np.where(converted, U, before)
    return before, after


def one_hot(classes: np.ndarray) -> ParcelGrid:
    composition = np.zeros((N_AGGREGATE_CLASSES,) + classes.shape, dtype=np.float32)
    for (row, col), value in np.ndenumerate(classes):
        if value >= 0:
            composition[value, row, col] = 1.0
    return ParcelGrid(composition)


class TestTheDefinition:
    def test_an_unchanged_landscape_scores_zero(self) -> None:
        before, _ = six_of_eight()
        assert c_align_from_classes(before, before) == 0.0
        assert not c_align_map(before, before).any()

    def test_six_of_eight_woodland_becoming_urban_scores_three_quarters(
        self,
    ) -> None:
        before, after = six_of_eight()
        assert c_align_map(before, after)[1, 1] == pytest.approx(0.75)

    def test_the_published_definition_scored_that_change_zero(self) -> None:
        """The difference between the two definitions, on the same neighbourhood."""
        before, after = six_of_eight()
        published = (commonality_map(before) - commonality_map(after)) ** 2
        assert published[1, 1] == pytest.approx(0.0)

    def test_a_neighbourhood_sharing_no_type_with_its_old_one_scores_one(
        self,
    ) -> None:
        before = np.full((3, 3), W, dtype=np.int64)
        before[1, 1] = A
        after = np.where(before == W, U, before)
        assert c_align_map(before, after)[1, 1] == pytest.approx(1.0)

    def test_border_parcels_count_only_their_in_grid_neighbours(self) -> None:
        before, after = six_of_eight()
        # (0, 0) sees (0, 1) W, (1, 0) W and (1, 1) A: two of three change type.
        assert c_align_map(before, after)[0, 0] == pytest.approx(2 / 3)

    def test_it_is_the_mean_over_every_parcel_of_the_grid(self) -> None:
        before, after = six_of_eight()
        assert c_align_from_classes(before, after) == pytest.approx(
            float(reference_terms(before, after).mean())
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_it_matches_a_naive_implementation_of_the_formula(self, seed: int) -> None:
        before, after = random_pair(np.random.default_rng(seed))
        np.testing.assert_allclose(
            c_align_map(before, after), reference_terms(before, after), atol=1e-12
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_it_is_bounded_and_symmetric(self, seed: int) -> None:
        before, after = random_pair(np.random.default_rng(100 + seed))
        value = c_align_from_classes(before, after)
        assert 0.0 <= value <= 1.0
        assert value == pytest.approx(c_align_from_classes(after, before))


class TestParcelsWithoutLandCover:
    def test_no_data_neighbours_are_left_out_of_the_shares(self) -> None:
        before = np.array([[A, A, NO_DATA]], dtype=np.int64)
        after = np.array([[U, A, NO_DATA]], dtype=np.int64)
        # (0, 1) has one land-covered neighbour, which changed type: 1, not 1/2.
        assert c_align_map(before, after)[0, 1] == pytest.approx(1.0)

    def test_a_parcel_with_no_land_covered_neighbour_contributes_zero(self) -> None:
        before = np.array([[A, NO_DATA]], dtype=np.int64)
        assert c_align_map(before, before)[0, 0] == 0.0

    def test_the_two_landscapes_must_cover_the_same_parcels(self) -> None:
        before = np.array([[A, NO_DATA]], dtype=np.int64)
        after = np.array([[A, U]], dtype=np.int64)
        with pytest.raises(ValueError, match="land-covered parcels"):
            c_align_from_classes(before, after)

    def test_shapes_must_match(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            c_align_from_classes(np.zeros((2, 2), np.int64), np.zeros((3, 3), np.int64))


class TestTheEntryPointsAgree:
    def test_the_parcel_grid_form_reads_the_dominant_classes(self) -> None:
        before, _ = six_of_eight()
        grid = one_hot(before)
        converted, _ = grid.convert_to_urban(
            [ParcelIndex(0, 0), ParcelIndex(2, 1)], fraction=1.0
        )
        assert c_align(grid, converted) == c_align_from_classes(
            grid.dominant_class(), converted.dominant_class()
        )

    def test_a_stored_record_grid_gives_the_same_number(self) -> None:
        """A run record stores dominant-class maps; scoring them is exact."""
        from chapter4_qolu.run_record import (
            class_grid,
            decode_class_grid,
            encode_class_grid,
        )

        before, after = random_pair(np.random.default_rng(7))
        grid, final = one_hot(before), one_hot(after)
        stored = [
            decode_class_grid(encode_class_grid(class_grid(item)))
            for item in (grid, final)
        ]
        assert c_align_from_classes(*stored) == c_align(grid, final)

    def test_the_radius_must_be_positive(self) -> None:
        before, after = six_of_eight()
        with pytest.raises(ValueError, match="radius"):
            c_align_map(before, after, radius=0)
