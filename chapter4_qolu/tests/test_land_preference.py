"""Tests for the land preference metric (Eq. 3 of the paper).

LP = sum_i w_i * P_phi_TD_i(L) + P_society(L)
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.config import SCENARIOS, ScenarioWeights
from chapter4_qolu.env.land_preference import (
    cooccurrence_matrix,
    land_preference,
    objective_preference,
    societal_preference,
)
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.env.rewards import TopDownObjective
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index


def two_class_grid() -> ParcelGrid:
    """A 1x3 strip: woodland | arable | woodland."""
    composition = np.zeros((N_AGGREGATE_CLASSES, 1, 3), dtype=np.float32)
    composition[band_index(AggregateClass.BROADLEAF_WOODLAND), 0, 0] = 1.0
    composition[band_index(AggregateClass.ARABLE), 0, 1] = 1.0
    composition[band_index(AggregateClass.BROADLEAF_WOODLAND), 0, 2] = 1.0
    return ParcelGrid(composition)


class TestCooccurrenceMatrix:
    def test_shape_and_row_stochasticity(self) -> None:
        matrix = cooccurrence_matrix(two_class_grid())
        assert matrix.shape == (N_AGGREGATE_CLASSES, N_AGGREGATE_CLASSES)
        populated = matrix.sum(axis=1) > 0
        np.testing.assert_allclose(matrix[populated].sum(axis=1), 1.0)

    def test_hand_computed_neighbourhood_cooccurrence(self) -> None:
        grid = two_class_grid()
        matrix = cooccurrence_matrix(grid, radius=1)
        woodland = band_index(AggregateClass.BROADLEAF_WOODLAND)
        arable = band_index(AggregateClass.ARABLE)
        # Woodland parcels (cols 0 and 2) each have exactly one neighbour: arable.
        assert matrix[woodland, arable] == pytest.approx(1.0)
        assert matrix[woodland, woodland] == pytest.approx(0.0)
        # The arable parcel has two woodland neighbours.
        assert matrix[arable, woodland] == pytest.approx(1.0)

    def test_absent_classes_have_empty_rows(self) -> None:
        matrix = cooccurrence_matrix(two_class_grid())
        np.testing.assert_allclose(matrix[band_index(AggregateClass.SALTWATER)], 0.0)

    def test_rejects_non_positive_radius(self) -> None:
        with pytest.raises(ValueError):
            cooccurrence_matrix(two_class_grid(), radius=0)


class TestSocietalPreference:
    def test_returns_the_row_for_the_focal_class(self) -> None:
        grid = two_class_grid()
        matrix = cooccurrence_matrix(grid)
        preference = societal_preference(matrix, AggregateClass.BROADLEAF_WOODLAND)
        np.testing.assert_allclose(
            preference, matrix[band_index(AggregateClass.BROADLEAF_WOODLAND)]
        )

    def test_marginal_distribution_when_no_focal_class_is_given(self) -> None:
        grid = two_class_grid()
        matrix = cooccurrence_matrix(grid)
        preference = societal_preference(matrix, None)
        assert preference.sum() == pytest.approx(1.0)


class TestObjectivePreference:
    def test_each_objective_maps_to_the_land_types_it_protects(self) -> None:
        land_use = objective_preference(TopDownObjective.LAND_USE_SEPARATION)
        woodland = objective_preference(TopDownObjective.WOODLAND_PRESERVATION)
        urban = objective_preference(TopDownObjective.URBAN_PROXIMITY)

        assert land_use[band_index(AggregateClass.MOUNTAIN_HEATH_BOG)] == pytest.approx(
            1.0
        )
        assert woodland[band_index(AggregateClass.BROADLEAF_WOODLAND)] > 0.0
        assert woodland[band_index(AggregateClass.ARABLE)] > 0.0
        assert urban[band_index(AggregateClass.BUILT_UP)] == pytest.approx(1.0)

    def test_preferences_are_probability_distributions(self) -> None:
        for objective in TopDownObjective:
            preference = objective_preference(objective)
            assert preference.shape == (N_AGGREGATE_CLASSES,)
            assert preference.sum() == pytest.approx(1.0)
            assert np.all(preference >= 0.0)


class TestLandPreferenceMetric:
    def test_matches_the_paper_formula(self) -> None:
        preferences = {
            TopDownObjective.LAND_USE_SEPARATION: np.array([1.0, 0.0, 0.0]),
            TopDownObjective.WOODLAND_PRESERVATION: np.array([0.0, 1.0, 0.0]),
            TopDownObjective.URBAN_PROXIMITY: np.array([0.0, 0.0, 1.0]),
        }
        weights = ScenarioWeights(land_use=0.5, woodland=0.25, urban=0.25)
        societal = np.array([0.2, 0.3, 0.5])
        metric = land_preference(preferences, weights, societal)
        np.testing.assert_allclose(metric, np.array([0.7, 0.55, 0.75]))

    def test_unnormalised_by_default_as_published(self) -> None:
        preferences = {
            objective: np.array([0.5, 0.5]) for objective in TopDownObjective
        }
        metric = land_preference(preferences, SCENARIOS[1], np.array([0.5, 0.5]))
        assert metric.sum() == pytest.approx(SCENARIOS[1].total() + 1.0)

    def test_optional_normalisation(self) -> None:
        preferences = {
            objective: np.array([0.5, 0.5]) for objective in TopDownObjective
        }
        metric = land_preference(
            preferences, SCENARIOS[1], np.array([0.25, 0.75]), normalise=True
        )
        assert metric.sum() == pytest.approx(1.0)

    def test_rejects_shape_mismatch(self) -> None:
        preferences = {
            objective: np.array([0.5, 0.5]) for objective in TopDownObjective
        }
        with pytest.raises(ValueError):
            land_preference(preferences, SCENARIOS[1], np.array([1.0, 0.0, 0.0]))

    def test_requires_all_objectives(self) -> None:
        with pytest.raises(KeyError):
            land_preference(
                {TopDownObjective.URBAN_PROXIMITY: np.array([1.0])},
                SCENARIOS[1],
                np.array([1.0]),
            )
