"""Tests for the dependency-free statistics behind the result tables."""

from __future__ import annotations

import math

import pytest

from chapter4_qolu.reporting.stats import (
    MIN_PAIRED_SAMPLES,
    cohens_d_paired,
    paired_t_test,
    regularised_incomplete_beta,
    summarise,
    two_sided_t_p_value,
)


class TestSummarise:
    def test_reports_mean_sd_and_standard_error(self) -> None:
        summary = summarise([1.0, 2.0, 3.0, 4.0, 5.0])
        assert summary.n == 5
        assert summary.mean == pytest.approx(3.0)
        # sample SD with ddof = 1: sqrt(10 / 4) = sqrt(2.5)
        assert summary.sd == pytest.approx(math.sqrt(2.5))
        assert summary.se == pytest.approx(math.sqrt(2.5) / math.sqrt(5))

    def test_single_value_has_no_spread(self) -> None:
        summary = summarise([0.7])
        assert summary.n == 1
        assert summary.mean == pytest.approx(0.7)
        assert summary.sd == 0.0
        assert summary.se == 0.0

    def test_rejects_an_empty_sample(self) -> None:
        with pytest.raises(ValueError):
            summarise([])


class TestIncompleteBeta:
    def test_matches_the_closed_form_for_a_equals_two_b_equals_half(self) -> None:
        # I_x(2, 1/2) = 1 - (3/2) sqrt(1-x) + (1/2) (1-x)^(3/2), integrated by
        # hand from B(2, 1/2) = 4/3 (see TestPairedTTest for the derivation).
        for x in (0.05, 0.25, 0.5, 0.75, 0.99):
            expected = 1.0 - 1.5 * math.sqrt(1.0 - x) + 0.5 * (1.0 - x) ** 1.5
            assert regularised_incomplete_beta(2.0, 0.5, x) == pytest.approx(expected)

    def test_is_zero_and_one_at_the_boundaries(self) -> None:
        assert regularised_incomplete_beta(2.0, 0.5, 0.0) == 0.0
        assert regularised_incomplete_beta(2.0, 0.5, 1.0) == 1.0

    def test_is_symmetric_for_equal_parameters(self) -> None:
        assert regularised_incomplete_beta(3.0, 3.0, 0.5) == pytest.approx(0.5)


class TestTwoSidedPValue:
    def test_a_zero_statistic_is_never_significant(self) -> None:
        assert two_sided_t_p_value(0.0, 4) == pytest.approx(1.0)

    def test_is_symmetric_in_the_sign_of_t(self) -> None:
        assert two_sided_t_p_value(2.5, 7) == pytest.approx(
            two_sided_t_p_value(-2.5, 7)
        )

    def test_decreases_as_the_statistic_grows(self) -> None:
        values = [two_sided_t_p_value(t, 4) for t in (0.5, 1.0, 2.0, 4.0, 8.0)]
        assert values == sorted(values, reverse=True)


class TestPairedTTest:
    """The reference cell is computed by hand.

    Differences ``d = [1, 2, 3, 4, 5]``: mean 3, sum of squared deviations
    ``4 + 1 + 0 + 1 + 4 = 10``, so ``sd = sqrt(10 / 4) = sqrt(2.5)`` and
    ``se = sqrt(2.5 / 5) = sqrt(0.5)``.  Hence ``t = 3 / sqrt(0.5) = 3 sqrt(2)``
    on ``df = 4``.

    The two-sided p-value is ``I_x(df/2, 1/2)`` at ``x = df / (df + t^2)``.
    With ``df = 4`` that is ``I_x(2, 1/2)``; ``B(2, 1/2) = 4/3`` and
    ``\\int_0^x u (1-u)^{-1/2} du = 4/3 - 2 sqrt(1-x) + (2/3)(1-x)^{3/2}``, so
    ``I_x(2, 1/2) = 1 - (3/2) s + (1/2) s^3`` with ``s = sqrt(1-x) =
    t / sqrt(4 + t^2) = 3 / sqrt(11)``.  That evaluates to 0.0132358...
    """

    treatment = (4.0, 5.0, 6.0, 7.0, 8.0)
    control = (3.0, 3.0, 3.0, 3.0, 3.0)

    def test_statistic_and_degrees_of_freedom_are_the_hand_computed_values(
        self,
    ) -> None:
        result = paired_t_test(self.treatment, self.control)
        assert result.possible is True
        assert result.n == 5
        assert result.df == 4
        assert result.mean_difference == pytest.approx(3.0)
        assert result.t == pytest.approx(3.0 * math.sqrt(2.0))

    def test_p_value_matches_the_hand_integrated_closed_form(self) -> None:
        result = paired_t_test(self.treatment, self.control)
        s = 3.0 / math.sqrt(11.0)
        expected = 1.0 - 1.5 * s + 0.5 * s**3
        assert result.p_value == pytest.approx(expected)
        assert result.p_value == pytest.approx(0.0132359, abs=1e-6)

    def test_cohens_d_is_the_paired_effect_size(self) -> None:
        result = paired_t_test(self.treatment, self.control)
        assert result.cohens_d == pytest.approx(3.0 / math.sqrt(2.5))
        assert cohens_d_paired(self.treatment, self.control) == pytest.approx(
            3.0 / math.sqrt(2.5)
        )

    def test_sign_follows_the_direction_of_the_difference(self) -> None:
        forward = paired_t_test(self.treatment, self.control)
        reverse = paired_t_test(self.control, self.treatment)
        assert reverse.t == pytest.approx(-(forward.t or 0.0))
        assert reverse.p_value == pytest.approx(forward.p_value)

    def test_fewer_than_three_pairs_is_reported_as_impossible(self) -> None:
        assert MIN_PAIRED_SAMPLES == 3
        result = paired_t_test((0.8, 0.9), (0.7, 0.6))
        assert result.possible is False
        assert result.t is None
        assert result.p_value is None
        assert result.n == 2
        assert "no paired t-test" in result.note

    def test_identical_samples_give_no_difference_at_all(self) -> None:
        result = paired_t_test((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert result.possible is True
        assert result.t == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)
        assert result.cohens_d == pytest.approx(0.0)

    def test_a_constant_non_zero_difference_is_perfectly_separated(self) -> None:
        result = paired_t_test((0.9, 0.9, 0.9), (0.5, 0.5, 0.5))
        assert result.possible is True
        assert result.t == math.inf
        assert result.p_value == pytest.approx(0.0)
        assert "zero variance" in result.note

    def test_rejects_unequal_sample_lengths(self) -> None:
        with pytest.raises(ValueError):
            paired_t_test((0.1, 0.2, 0.3), (0.1, 0.2))


class TestAgainstScipy:
    """Cross-check against SciPy where it happens to be installed.

    The package itself never imports SciPy - :mod:`chapter4_qolu.reporting.stats`
    is numpy-only - so this test is skipped rather than required.
    """

    def test_matches_scipy_paired_t_test(self) -> None:
        stats = pytest.importorskip("scipy.stats")
        treatment = [0.81, 0.77, 0.90, 0.84, 0.79, 0.88]
        control = [0.70, 0.74, 0.69, 0.80, 0.71, 0.73]
        reference = stats.ttest_rel(treatment, control)
        result = paired_t_test(treatment, control)
        assert result.t == pytest.approx(float(reference.statistic))
        assert result.p_value == pytest.approx(float(reference.pvalue))
