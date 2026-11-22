"""Unit tests for the Q-distribution diagnostics.

Every statistic is checked on a synthetic categorical distribution whose answer
can be written down by hand, so a regression in the diagnostic cannot be
mistaken for a finding about the agent.
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.diagnostics.calibration import (
    atom_indices,
    interval_coverage,
    kolmogorov_smirnov_uniform,
    out_of_support_fraction,
    pit_histogram,
    predictive_quantile,
    randomised_pit,
)
from chapter4_qolu.diagnostics.crps import (
    climatology_probabilities,
    crps,
    crps_skill_score,
)
from chapter4_qolu.diagnostics.errors import (
    BOUNDARY_MASS_THRESHOLD,
    boundary_mass,
    error_statistics,
    predicted_means,
    support_resolution,
)
from chapter4_qolu.diagnostics.returns import (
    n_step_reward_sums,
    n_step_targets,
    tail_returns,
)

ATOMS3 = np.array([-1.0, 0.0, 1.0])
ATOMS5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


class TestRealisedReturns:
    def test_the_tail_return_is_the_backward_discounted_sum(self) -> None:
        returns = tail_returns(np.array([1.0, 2.0, 3.0]), gamma=0.5)
        np.testing.assert_allclose(returns, [2.75, 3.5, 3.0])

    def test_the_tail_return_of_a_single_reward_is_that_reward(self) -> None:
        np.testing.assert_allclose(tail_returns(np.array([4.0]), gamma=0.9), [4.0])

    def test_the_n_step_reward_sum_truncates_at_the_end_of_the_run(self) -> None:
        sums = n_step_reward_sums(np.array([1.0, 2.0, 3.0]), gamma=0.5, n_step=2)
        np.testing.assert_allclose(sums, [2.0, 3.5, 3.0])

    def test_the_n_step_target_adds_the_agents_own_bootstrap(self) -> None:
        targets = n_step_targets(
            np.array([1.0, 2.0, 3.0]),
            gamma=0.5,
            n_step=2,
            bootstrap_values=np.array([10.0, 20.0, 30.0]),
        )
        np.testing.assert_allclose(targets, [7.0, 11.0, 18.0])

    def test_a_one_step_target_is_the_ordinary_td_target(self) -> None:
        targets = n_step_targets(
            np.array([1.0, 2.0]),
            gamma=0.9,
            n_step=1,
            bootstrap_values=np.array([5.0, 7.0]),
        )
        np.testing.assert_allclose(targets, [1.0 + 0.9 * 5.0, 2.0 + 0.9 * 7.0])

    def test_returns_reject_a_non_one_dimensional_reward_stream(self) -> None:
        with pytest.raises(ValueError):
            tail_returns(np.zeros((2, 2)), gamma=0.9)


class TestPointErrors:
    def test_the_predicted_mean_is_the_atom_weighted_average(self) -> None:
        probabilities = np.array([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
        np.testing.assert_allclose(predicted_means(probabilities, ATOMS3), [0.0, 0.0])

    def test_mae_and_rmse_of_a_point_mass_predictor(self) -> None:
        probabilities = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        statistics = error_statistics(probabilities, ATOMS3, np.array([2.0, -4.0]))
        assert statistics["mae"] == pytest.approx(3.0)
        assert statistics["rmse"] == pytest.approx(np.sqrt((4.0 + 16.0) / 2))
        assert statistics["bias"] == pytest.approx(1.0)
        assert statistics["n"] == 2

    def test_a_perfect_mean_predictor_scores_zero(self) -> None:
        probabilities = np.array([[0.0, 0.0, 1.0]])
        statistics = error_statistics(probabilities, ATOMS3, np.array([1.0]))
        assert statistics["mae"] == pytest.approx(0.0)
        assert statistics["rmse"] == pytest.approx(0.0)


class TestBoundaryMass:
    def test_it_separates_the_two_extreme_atoms(self) -> None:
        probabilities = np.array([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
        summary = boundary_mass(probabilities)
        assert summary["lower_mean"] == pytest.approx(0.25)
        assert summary["upper_mean"] == pytest.approx(0.25)
        assert summary["total_mean"] == pytest.approx(0.5)
        assert summary["total_max"] == pytest.approx(1.0)

    def test_the_flagged_fraction_counts_records_over_the_threshold(self) -> None:
        probabilities = np.array([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.99, 0.01]])
        summary = boundary_mass(probabilities)
        assert summary["threshold"] == pytest.approx(BOUNDARY_MASS_THRESHOLD)
        assert summary["fraction_above_threshold"] == pytest.approx(1.0 / 3.0)

    def test_a_distribution_with_no_edge_mass_reports_zero(self) -> None:
        summary = boundary_mass(np.array([[0.0, 1.0, 0.0]]))
        assert summary["total_mean"] == pytest.approx(0.0)
        assert summary["flagged"] is False


class TestSupportResolution:
    """The other C51 support pathology: a support far too *wide* for the signal."""

    PAPER_SUPPORT = np.linspace(-200.0, 200.0, 51)

    def test_delta_z_is_the_published_atom_spacing(self) -> None:
        summary = support_resolution(self.PAPER_SUPPORT, np.array([0.0, 1.0]))
        assert summary["delta_z"] == pytest.approx(8.0)
        assert summary["n_atoms"] == 51

    def test_a_signal_finer_than_one_atom_is_flagged(self) -> None:
        rng = np.random.default_rng(0)
        realised = rng.normal(0.0, 1.0, size=500)
        summary = support_resolution(self.PAPER_SUPPORT, realised)
        assert summary["realised_sd_in_atoms"] < 1.0
        assert summary["resolves_the_signal"] is False
        assert summary["distinct_atoms_used"] <= 2

    def test_a_signal_spanning_many_atoms_is_not_flagged(self) -> None:
        rng = np.random.default_rng(0)
        realised = rng.normal(0.0, 60.0, size=500)
        summary = support_resolution(self.PAPER_SUPPORT, realised)
        assert summary["realised_sd_in_atoms"] > 1.0
        assert summary["resolves_the_signal"] is True
        assert summary["distinct_atoms_used"] > 2

    def test_distinct_atoms_used_counts_the_cells_the_returns_land_in(self) -> None:
        summary = support_resolution(ATOMS3, np.array([-1.0, -0.9, 0.0, 1.0]))
        assert summary["distinct_atoms_used"] == 3


class TestAtomBinning:
    def test_a_realised_value_snaps_to_the_nearest_atom(self) -> None:
        indices = atom_indices(ATOMS3, np.array([-0.6, -0.4, 0.4, 0.6]))
        np.testing.assert_array_equal(indices, [0, 1, 1, 2])

    def test_values_beyond_the_support_are_clamped_onto_it(self) -> None:
        indices = atom_indices(ATOMS3, np.array([-99.0, 99.0]))
        np.testing.assert_array_equal(indices, [0, 2])

    def test_the_out_of_support_fraction_is_reported(self) -> None:
        fraction = out_of_support_fraction(ATOMS3, np.array([-99.0, 0.0, 0.5, 99.0]))
        assert fraction == pytest.approx(0.5)


class TestRandomisedPIT:
    def test_the_transform_is_f_below_plus_v_times_the_atom_mass(self) -> None:
        probabilities = np.array([[0.2, 0.3, 0.5]])
        values = randomised_pit(
            probabilities, ATOMS3, np.array([0.1]), np.random.default_rng(0)
        )
        draw = np.random.default_rng(0).random(1)
        np.testing.assert_allclose(values, 0.2 + 0.3 * draw)

    def test_a_point_mass_hit_exactly_returns_the_randomisation_draw(self) -> None:
        probabilities = np.array([[0.0, 1.0, 0.0]])
        values = randomised_pit(
            probabilities, ATOMS3, np.array([0.0]), np.random.default_rng(7)
        )
        np.testing.assert_allclose(values, np.random.default_rng(7).random(1))

    def test_an_observation_above_all_the_mass_transforms_to_one(self) -> None:
        probabilities = np.array([[1.0, 0.0, 0.0]])
        values = randomised_pit(
            probabilities, ATOMS3, np.array([1.0]), np.random.default_rng(0)
        )
        np.testing.assert_allclose(values, [1.0])

    def test_a_perfectly_calibrated_forecast_is_uniform(self) -> None:
        rng = np.random.default_rng(20260929)
        atoms = np.linspace(-5.0, 5.0, 11)
        weights = rng.dirichlet(np.ones(11), size=4000)
        draws = np.array(
            [atoms[rng.choice(11, p=row)] for row in weights], dtype=np.float64
        )
        values = randomised_pit(weights, atoms, draws, rng)
        statistic, p_value = kolmogorov_smirnov_uniform(values)
        assert statistic < 0.05
        assert p_value > 0.05

    def test_an_over_confident_forecast_is_detected(self) -> None:
        rng = np.random.default_rng(11)
        atoms = np.linspace(-5.0, 5.0, 11)
        probabilities = np.zeros((2000, 11))
        probabilities[:, 5] = 1.0
        draws = atoms[rng.integers(0, 11, size=2000)]
        statistic, p_value = kolmogorov_smirnov_uniform(
            randomised_pit(probabilities, atoms, draws, rng)
        )
        assert statistic > 0.3
        assert p_value < 0.01


class TestKolmogorovSmirnov:
    def test_a_single_point_gives_the_hand_computed_statistic(self) -> None:
        statistic, _ = kolmogorov_smirnov_uniform(np.array([0.5]))
        assert statistic == pytest.approx(0.5)

    def test_three_evenly_spaced_points_give_one_quarter(self) -> None:
        statistic, _ = kolmogorov_smirnov_uniform(np.array([0.25, 0.5, 0.75]))
        assert statistic == pytest.approx(0.25)

    def test_the_p_value_decreases_with_the_statistic(self) -> None:
        _, close = kolmogorov_smirnov_uniform(np.linspace(0.005, 0.995, 100))
        _, far = kolmogorov_smirnov_uniform(np.linspace(0.0, 0.2, 100))
        assert close > far

    def test_an_empty_sample_is_refused(self) -> None:
        with pytest.raises(ValueError):
            kolmogorov_smirnov_uniform(np.array([]))


class TestPITHistogram:
    def test_a_uniform_sample_has_unit_density_in_every_bin(self) -> None:
        histogram = pit_histogram(np.linspace(0.0005, 0.9995, 2000), bins=10)
        np.testing.assert_allclose(histogram["density"], np.ones(10), atol=1e-6)
        assert len(histogram["edges"]) == 11

    def test_the_counts_add_up_to_the_sample_size(self) -> None:
        histogram = pit_histogram(np.array([0.05, 0.15, 0.95]), bins=10)
        assert sum(histogram["counts"]) == 3


class TestIntervalCoverage:
    def test_the_predictive_quantile_is_the_generalised_inverse_cdf(self) -> None:
        probabilities = np.full((1, 5), 0.2)
        assert predictive_quantile(probabilities, ATOMS5, 0.25)[0] == pytest.approx(
            -1.0
        )
        assert predictive_quantile(probabilities, ATOMS5, 0.75)[0] == pytest.approx(1.0)
        assert predictive_quantile(probabilities, ATOMS5, 0.05)[0] == pytest.approx(
            -2.0
        )

    def test_coverage_counts_the_observations_inside_the_interval(self) -> None:
        probabilities = np.full((5, 5), 0.2)
        realised = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        rows = {
            row["level"]: row
            for row in interval_coverage(probabilities, ATOMS5, realised)
        }
        assert rows[0.5]["empirical"] == pytest.approx(0.6)
        assert rows[0.5]["nominal"] == pytest.approx(0.5)
        assert rows[0.9]["empirical"] == pytest.approx(1.0)
        assert rows[0.5]["mean_width"] == pytest.approx(2.0)

    def test_a_point_mass_covers_only_an_exact_hit(self) -> None:
        probabilities = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        rows = {
            row["level"]: row
            for row in interval_coverage(probabilities, ATOMS3, np.array([0.0, 1.0]))
        }
        assert rows[0.9]["empirical"] == pytest.approx(0.5)


class TestCRPS:
    def test_a_point_mass_scores_the_absolute_error(self) -> None:
        probabilities = np.array([[0.0, 1.0, 0.0]])
        np.testing.assert_allclose(crps(probabilities, ATOMS3, np.array([2.5])), [2.5])

    def test_a_two_point_forecast_matches_the_energy_form(self) -> None:
        probabilities = np.array([[0.5, 0.0, 0.5]])
        np.testing.assert_allclose(crps(probabilities, ATOMS3, np.array([0.0])), [0.5])

    def test_it_matches_a_numerical_integration_of_the_definition(self) -> None:
        rng = np.random.default_rng(3)
        atoms = np.linspace(-4.0, 4.0, 9)
        probabilities = rng.dirichlet(np.ones(9), size=1)
        realised = np.array([0.7])
        grid = np.linspace(-10.0, 10.0, 200_001)
        cumulative = np.concatenate([[0.0], np.cumsum(probabilities[0])])
        cdf = cumulative[np.searchsorted(atoms, grid, side="right")]
        indicator = (grid >= realised[0]).astype(float)
        expected = float(np.trapezoid((cdf - indicator) ** 2, grid))
        np.testing.assert_allclose(
            crps(probabilities, atoms, realised), [expected], rtol=1e-3
        )

    def test_a_sharper_correct_forecast_scores_better(self) -> None:
        sharp = np.array([[0.0, 1.0, 0.0]])
        vague = np.array([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
        realised = np.array([0.0])
        assert crps(sharp, ATOMS3, realised)[0] < crps(vague, ATOMS3, realised)[0]


class TestClimatologyAndSkill:
    def test_the_climatology_is_the_marginal_histogram_of_the_run(self) -> None:
        realised = np.array([-1.0, -1.0, 0.0, 1.0])
        np.testing.assert_allclose(
            climatology_probabilities(ATOMS3, realised), [0.5, 0.25, 0.25]
        )

    def test_the_climatology_sums_to_one(self) -> None:
        realised = np.array([-99.0, 0.3, 42.0])
        assert climatology_probabilities(ATOMS3, realised).sum() == pytest.approx(1.0)

    def test_the_skill_score_is_one_minus_the_ratio(self) -> None:
        assert crps_skill_score(1.0, 4.0) == pytest.approx(0.75)
        assert crps_skill_score(4.0, 4.0) == pytest.approx(0.0)
        assert crps_skill_score(8.0, 4.0) == pytest.approx(-1.0)

    def test_a_zero_reference_has_no_defined_skill(self) -> None:
        assert np.isnan(crps_skill_score(0.5, 0.0))

    def test_a_state_conditional_forecast_beats_its_own_climatology(self) -> None:
        atoms = np.array([-1.0, 0.0, 1.0])
        realised = np.array([-1.0, 1.0] * 50)
        probabilities = np.zeros((100, 3))
        probabilities[0::2, 0] = 1.0  # exactly right, conditioned on the state
        probabilities[1::2, 2] = 1.0
        model = float(crps(probabilities, atoms, realised).mean())
        climatology = np.repeat(
            climatology_probabilities(atoms, realised)[None, :], 100, axis=0
        )
        reference = float(crps(climatology, atoms, realised).mean())
        assert crps_skill_score(model, reference) > 0.0

    def test_a_state_blind_forecast_has_no_skill_over_climatology(self) -> None:
        atoms = np.array([-1.0, 0.0, 1.0])
        realised = np.array([-1.0, 1.0] * 50)
        climatology = np.repeat(
            climatology_probabilities(atoms, realised)[None, :], 100, axis=0
        )
        reference = float(crps(climatology, atoms, realised).mean())
        assert crps_skill_score(reference, reference) == pytest.approx(0.0)
