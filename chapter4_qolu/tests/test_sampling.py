"""Tests for the attractiveness-weighted decision buffer (Eq. 2 of the paper).

P(sampled | A_i) is proportional to A_i and 50 parcels are drawn *with
replacement* per iteration.
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.env.parcel_grid import ParcelIndex
from chapter4_qolu.env.sampling import (
    attractiveness_probabilities,
    sample_decision_buffer,
)


class TestAttractivenessProbabilities:
    def test_probabilities_are_proportional_to_attractiveness(self) -> None:
        scores = np.array([[1.0, 3.0], [0.0, 4.0]])
        probabilities = attractiveness_probabilities(scores)
        np.testing.assert_allclose(
            probabilities, np.array([[0.125, 0.375], [0.0, 0.5]])
        )

    def test_probabilities_sum_to_one(self) -> None:
        rng = np.random.default_rng(7)
        probabilities = attractiveness_probabilities(rng.random((6, 5)))
        assert probabilities.sum() == pytest.approx(1.0)

    def test_ineligible_parcels_receive_zero_probability(self) -> None:
        scores = np.array([[1.0, 1.0], [1.0, 1.0]])
        eligible = np.array([[True, False], [False, True]])
        probabilities = attractiveness_probabilities(scores, eligible=eligible)
        np.testing.assert_allclose(probabilities, np.array([[0.5, 0.0], [0.0, 0.5]]))

    def test_uniform_fallback_when_all_eligible_scores_are_zero(self) -> None:
        scores = np.zeros((2, 2))
        eligible = np.array([[True, True], [False, False]])
        probabilities = attractiveness_probabilities(scores, eligible=eligible)
        np.testing.assert_allclose(probabilities, np.array([[0.5, 0.5], [0.0, 0.0]]))

    def test_negative_scores_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            attractiveness_probabilities(np.array([[-1.0, 1.0]]))

    def test_no_eligible_parcels_raises(self) -> None:
        with pytest.raises(ValueError):
            attractiveness_probabilities(
                np.ones((2, 2)), eligible=np.zeros((2, 2), bool)
            )


class TestDecisionBufferSampling:
    def test_draws_the_requested_number_with_replacement(self) -> None:
        scores = np.ones((3, 3))
        parcels = sample_decision_buffer(
            scores, n_samples=50, rng=np.random.default_rng(0)
        )
        assert len(parcels) == 50
        assert len(set(parcels)) < 50  # 9 parcels, 50 draws -> repeats guaranteed
        assert all(isinstance(parcel, ParcelIndex) for parcel in parcels)

    def test_empirical_frequencies_match_the_attractiveness_weights(self) -> None:
        scores = np.array([[1.0, 3.0], [2.0, 4.0]])
        draws = sample_decision_buffer(
            scores, n_samples=200_000, rng=np.random.default_rng(42)
        )
        counts = np.zeros((2, 2))
        for parcel in draws:
            counts[parcel.row, parcel.col] += 1
        empirical = counts / counts.sum()
        np.testing.assert_allclose(empirical, scores / scores.sum(), atol=5e-3)

    def test_zero_attractiveness_parcels_are_never_drawn(self) -> None:
        scores = np.array([[0.0, 1.0], [0.0, 1.0]])
        draws = sample_decision_buffer(
            scores, n_samples=500, rng=np.random.default_rng(3)
        )
        assert all(parcel.col == 1 for parcel in draws)

    def test_respects_the_eligibility_mask(self) -> None:
        scores = np.ones((4, 4))
        eligible = np.zeros((4, 4), dtype=bool)
        eligible[2, 3] = True
        draws = sample_decision_buffer(
            scores, n_samples=20, rng=np.random.default_rng(1), eligible=eligible
        )
        assert set(draws) == {ParcelIndex(2, 3)}

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        scores = np.arange(16, dtype=float).reshape(4, 4)
        first = sample_decision_buffer(
            scores, n_samples=30, rng=np.random.default_rng(5)
        )
        second = sample_decision_buffer(
            scores, n_samples=30, rng=np.random.default_rng(5)
        )
        assert first == second

    def test_rejects_non_positive_sample_count(self) -> None:
        with pytest.raises(ValueError):
            sample_decision_buffer(
                np.ones((2, 2)), n_samples=0, rng=np.random.default_rng(0)
            )
