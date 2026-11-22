"""Tests for the MitM baseline (Bone, Dragicevic & White 2011)."""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.baselines import mitm as mitm_module
from chapter4_qolu.baselines.base import run_baseline
from chapter4_qolu.baselines.config import MitMConfig
from chapter4_qolu.baselines.mitm import MitMMethod, MitMStakeholder, learned_mitm
from chapter4_qolu.baselines.mitm_allocation import (
    compromise_allocation,
    compromise_scores,
    criterion_matrix,
    density_discount,
    greedy_choice,
    white_engelen_attractivity,
)
from chapter4_qolu.env.compliance import parcel_compliance
from chapter4_qolu.env.parcel_grid import ParcelIndex
from chapter4_qolu.env.urban_env import APPROVE
from chapter4_qolu.objectives import TopDownObjective
from chapter4_qolu.tests.baseline_fixtures import make_env


def make_method(env, seed: int = 0, **overrides: object) -> MitMMethod:
    return MitMMethod.for_env(env, seed=seed, **overrides)  # type: ignore[arg-type]


class TestNonNeural:
    """MitM is a 2011 GIS/ABM method; upgrading it to a network would
    silently strengthen the baseline and misrepresent the comparison."""

    def test_the_module_does_not_use_torch(self) -> None:
        assert not hasattr(mitm_module, "torch")
        assert not hasattr(mitm_module, "nn")

    def test_the_method_holds_only_tabular_state(self) -> None:
        method = make_method(make_env())
        for stakeholder in method.stakeholders:
            assert isinstance(stakeholder, MitMStakeholder)
            assert isinstance(stakeholder.value, float)
            assert stakeholder.criterion_weights.shape == (3,)


class TestWhiteEngelenAttractivity:
    def test_follows_the_published_multiplicative_form(self) -> None:
        scores = white_engelen_attractivity(
            suitability=np.array([0.5, 1.0]),
            neighbourhood=np.array([0.25, 0.0]),
            neighbourhood_weight=2.0,
            perturbation=np.array([0.0, 0.5]),
        )
        np.testing.assert_allclose(scores, [0.5 * 1.5, 1.0 * 1.0 * 1.5])

    def test_zero_suitability_kills_the_cell(self) -> None:
        scores = white_engelen_attractivity(
            suitability=np.zeros(3),
            neighbourhood=np.ones(3),
            neighbourhood_weight=1.0,
            perturbation=np.ones(3),
        )
        np.testing.assert_allclose(scores, 0.0)

    def test_is_monotone_in_the_neighbourhood_term(self) -> None:
        low, high = white_engelen_attractivity(
            suitability=np.array([1.0, 1.0]),
            neighbourhood=np.array([0.1, 0.9]),
            neighbourhood_weight=1.0,
            perturbation=np.zeros(2),
        )
        assert high > low


class TestCriterionMatrix:
    def test_one_column_per_published_objective(self) -> None:
        env = make_env()
        batch = env.reset(seed=0)
        criteria = criterion_matrix(env.grid, batch.parcels, env.config.reward)
        assert criteria.shape == (len(batch.parcels), 3)
        assert ((criteria >= 0.0) & (criteria <= 1.0)).all()

    def test_columns_are_the_published_per_parcel_compliance_scores(self) -> None:
        env = make_env()
        batch = env.reset(seed=1)
        criteria = criterion_matrix(env.grid, batch.parcels, env.config.reward)
        for column, objective in enumerate(TopDownObjective):
            for row, parcel in enumerate(batch.parcels):
                assert criteria[row, column] == pytest.approx(
                    parcel_compliance(env.grid, parcel, objective, env.config.reward)
                )


class TestDensityDiscount:
    def test_a_crowded_neighbourhood_is_discounted(self) -> None:
        env = make_env()
        env.reset(seed=0)
        discount = density_discount(
            env.grid, (ParcelIndex(0, 0), ParcelIndex(6, 6)), pressure=1.0
        )
        assert discount[0] < discount[1]

    def test_zero_pressure_removes_the_discount(self) -> None:
        env = make_env()
        env.reset(seed=0)
        discount = density_discount(
            env.grid, (ParcelIndex(0, 0), ParcelIndex(6, 6)), pressure=0.0
        )
        np.testing.assert_allclose(discount, 1.0)


class TestGreedyChooser:
    def test_takes_the_highest_utility_cells_in_order(self) -> None:
        parcels = (ParcelIndex(0, 0), ParcelIndex(1, 1), ParcelIndex(2, 2))
        chosen = greedy_choice(np.array([0.1, 0.9, 0.5]), parcels, 2)
        assert chosen == (ParcelIndex(1, 1), ParcelIndex(2, 2))

    def test_never_returns_more_than_the_candidates(self) -> None:
        parcels = (ParcelIndex(0, 0),)
        assert len(greedy_choice(np.array([1.0]), parcels, 5)) == 1

    def test_is_myopic_and_does_not_learn(self) -> None:
        parcels = (ParcelIndex(0, 0), ParcelIndex(1, 1))
        utility = np.array([0.2, 0.8])
        first = greedy_choice(utility, parcels, 1)
        second = greedy_choice(utility, parcels, 1)
        assert first == second == (ParcelIndex(1, 1),)


class TestCompromiseProgramming:
    def test_the_ideal_alternative_scores_zero(self) -> None:
        criteria = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        scores = compromise_scores(criteria, np.array([1 / 3, 1 / 3, 1 / 3]))
        assert scores[0] == pytest.approx(0.0)
        assert scores[1] > scores[2] > scores[0]

    def test_weights_steer_which_criterion_dominates(self) -> None:
        criteria = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
        land_focused = compromise_scores(criteria, np.array([1.0, 0.0, 0.0]))
        assert land_focused[1] < land_focused[0]

    def test_a_criterion_with_no_spread_contributes_nothing(self) -> None:
        criteria = np.array([[0.4, 1.0], [0.4, 0.0]])
        scores = compromise_scores(criteria, np.array([0.5, 0.5]))
        assert scores[0] == pytest.approx(0.0)

    def test_allocation_takes_the_best_b_alternatives(self) -> None:
        parcels = (ParcelIndex(0, 0), ParcelIndex(1, 1), ParcelIndex(2, 2))
        criteria = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5]])
        chosen = compromise_allocation(
            parcels, criteria, np.array([1 / 3, 1 / 3, 1 / 3]), budget=2
        )
        assert chosen == (ParcelIndex(1, 1), ParcelIndex(2, 2))

    def test_an_absent_budget_allocates_everything(self) -> None:
        parcels = (ParcelIndex(0, 0), ParcelIndex(1, 1))
        criteria = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
        chosen = compromise_allocation(
            parcels, criteria, np.array([1 / 3, 1 / 3, 1 / 3]), budget=None
        )
        assert set(chosen) == set(parcels)

    def test_the_compromise_order_is_configurable(self) -> None:
        criteria = np.array([[0.0, 1.0, 1.0], [0.6, 0.6, 0.6]])
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
        euclidean = compromise_scores(criteria, weights, order=2.0)
        chebyshev = compromise_scores(criteria, weights, order=20.0)
        assert not np.allclose(euclidean, chebyshev)


class TestMitMConfig:
    def test_the_headline_variant_freezes_the_criterion_weights(self) -> None:
        assert MitMConfig().learn_weights is False

    def test_compromise_programming_defaults_to_the_euclidean_order(self) -> None:
        assert MitMConfig().compromise_order == 2.0

    def test_rejects_a_negative_learning_rate(self) -> None:
        with pytest.raises(ValueError):
            MitMConfig(value_learning_rate=-0.1)

    def test_rejects_a_proposal_multiplier_below_one(self) -> None:
        with pytest.raises(ValueError):
            MitMConfig(proposal_multiplier=0)


class TestFrozenWeights:
    def test_the_aggregate_criterion_weights_are_the_scenario_weights(self) -> None:
        for scenario in (1, 2, 3, 4):
            env = make_env(scenario=scenario)
            method = make_method(env)
            method.reset(env, 0)
            expected = env.weights.normalised()
            np.testing.assert_allclose(
                method.criterion_weights,
                [expected.land_use, expected.woodland, expected.urban],
                atol=1e-9,
            )

    def test_each_stakeholder_starts_on_its_own_criterion(self) -> None:
        env = make_env()
        method = make_method(env)
        method.reset(env, 0)
        for index, stakeholder in enumerate(method.stakeholders):
            expected = np.zeros(3)
            expected[index] = 1.0
            np.testing.assert_allclose(stakeholder.criterion_weights, expected)

    def test_weights_never_move_in_the_headline_variant(self) -> None:
        env = make_env()
        method = make_method(env)
        run_baseline(env, method, iterations=10, seed=0)
        for index, stakeholder in enumerate(method.stakeholders):
            expected = np.zeros(3)
            expected[index] = 1.0
            np.testing.assert_allclose(stakeholder.criterion_weights, expected)

    def test_the_values_still_track_the_realised_rewards(self) -> None:
        env = make_env()
        method = make_method(env)
        run_baseline(env, method, iterations=10, seed=0)
        assert any(stakeholder.value != 0.0 for stakeholder in method.stakeholders)


class TestLearnedWeights:
    def test_reports_a_distinct_name(self) -> None:
        assert learned_mitm(make_env(), seed=0).name == "mitm_learned"

    def test_the_headline_variant_keeps_the_published_name(self) -> None:
        assert make_method(make_env()).name == "mitm"

    def test_criterion_weights_move_away_from_the_scenario_weights(self) -> None:
        env = make_env()
        method = learned_mitm(env, seed=0, weight_learning_rate=0.5)
        before = method.criterion_weights.copy()
        run_baseline(env, method, iterations=20, seed=0)
        assert not np.allclose(before, method.criterion_weights)

    def test_criterion_weights_stay_on_the_simplex(self) -> None:
        env = make_env()
        method = learned_mitm(env, seed=0, weight_learning_rate=0.5)
        run_baseline(env, method, iterations=20, seed=0)
        for stakeholder in method.stakeholders:
            assert float(stakeholder.criterion_weights.sum()) == pytest.approx(1.0)
            assert (stakeholder.criterion_weights >= 0.0).all()

    def test_a_td_error_is_reported_as_the_learning_signal(self) -> None:
        env = make_env()
        report = run_baseline(env, learned_mitm(env, seed=0), iterations=10, seed=0)
        assert len(report.losses) == 10
        assert all(value >= 0.0 for value in report.losses)


class TestTabularTDLearner:
    def test_the_value_moves_a_fraction_of_the_td_error(self) -> None:
        stakeholder = MitMStakeholder(
            objective=TopDownObjective.WOODLAND_PRESERVATION,
            criterion_weights=np.array([0.0, 1.0, 0.0]),
        )
        error = stakeholder.observe(reward=1.0, rate=0.5)
        assert error == pytest.approx(1.0)
        assert stakeholder.value == pytest.approx(0.5)
        stakeholder.observe(reward=1.0, rate=0.5)
        assert stakeholder.value == pytest.approx(0.75)

    def test_a_better_than_expected_round_pulls_the_weights_toward_it(self) -> None:
        stakeholder = MitMStakeholder(
            objective=TopDownObjective.LAND_USE_SEPARATION,
            criterion_weights=np.array([1.0, 0.0, 0.0]),
        )
        stakeholder.nudge(error=1.0, realised=np.array([0.0, 1.0, 0.0]), rate=0.5)
        assert stakeholder.criterion_weights[1] > 0.0
        assert float(stakeholder.criterion_weights.sum()) == pytest.approx(1.0)

    def test_a_worse_than_expected_round_pushes_the_weights_away(self) -> None:
        stakeholder = MitMStakeholder(
            objective=TopDownObjective.LAND_USE_SEPARATION,
            criterion_weights=np.array([0.5, 0.5, 0.0]),
        )
        stakeholder.nudge(error=-1.0, realised=np.array([0.0, 1.0, 0.0]), rate=0.5)
        assert stakeholder.criterion_weights[1] < 0.5


class TestMitMMethod:
    def test_decisions_cover_every_candidate(self) -> None:
        env = make_env()
        method = make_method(env)
        batch = env.reset(seed=2)
        method.reset(env, 2)
        decision = method.decide(batch)
        assert decision.decisions.shape == (len(batch.parcels),)

    def test_the_bottom_up_layer_proposes_a_subset_of_the_candidates(self) -> None:
        env = make_env()
        method = make_method(env)
        batch = env.reset(seed=3)
        method.reset(env, 3)
        decision = method.decide(batch)
        assert decision.proposals is not None
        assert set(decision.proposals) <= set(batch.parcels)
        assert len(decision.proposals) <= len(set(batch.parcels))

    def test_approvals_are_a_subset_of_the_proposals(self) -> None:
        env = make_env()
        method = make_method(env)
        batch = env.reset(seed=4)
        method.reset(env, 4)
        decision = method.decide(batch)
        approved = {
            parcel
            for parcel, action in zip(batch.parcels, decision.decisions)
            if action == APPROVE
        }
        assert approved <= set(decision.proposals or ())

    def test_never_allocates_more_than_the_shared_conversion_budget(self) -> None:
        env = make_env(max_conversions_per_iteration=2)
        report = run_baseline(env, make_method(env), iterations=8, seed=5)
        assert max(report.n_conversions) <= 2

    def test_allocates_the_best_compromise_parcel_on_a_tiny_grid(self) -> None:
        env = make_env(
            grid_size=8, samples_per_iteration=10, max_conversions_per_iteration=1
        )
        method = make_method(env, perturbation_scale=0.0, proposal_multiplier=4)
        batch = env.reset(seed=6)
        method.reset(env, 6)
        decision = method.decide(batch)
        proposals = list(decision.proposals or ())
        criteria = criterion_matrix(env.grid, proposals, env.config.reward)
        scores = compromise_scores(criteria, method.criterion_weights)
        approved = [
            parcel
            for parcel, action in zip(batch.parcels, decision.decisions)
            if action == APPROVE
        ]
        assert len(set(approved)) == 1
        assert scores[proposals.index(approved[0])] == pytest.approx(scores.min())

    def test_ties_are_broken_by_parcel_order_so_runs_stay_deterministic(self) -> None:
        parcels = (ParcelIndex(2, 2), ParcelIndex(1, 1))
        criteria = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
        chosen = compromise_allocation(
            parcels, criteria, np.array([0.5, 0.5, 0.0]), budget=1
        )
        assert chosen == (ParcelIndex(1, 1),)

    def test_runs_end_to_end_and_reports_the_published_metrics(self) -> None:
        env = make_env()
        report = run_baseline(env, make_method(env), iterations=10, seed=7)
        assert report.method == "mitm"
        assert len(report.team_rewards) == 10
        assert set(report.final_metrics) == {
            "land_use",
            "woodland",
            "urban",
            "conversion_alignment",
            "c_align",
            "well_being",
        }

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        def run() -> tuple[float, ...]:
            env = make_env()
            return run_baseline(env, make_method(env, seed=8), 10, seed=8).team_rewards

        np.testing.assert_allclose(run(), run())

    def test_the_stochastic_perturbation_changes_what_is_chosen(self) -> None:
        # The perturbation only bites when the bottom-up layer is selective,
        # i.e. when proposal_multiplier * B is smaller than the candidate pool.
        def run(scale: float) -> tuple[float, ...]:
            env = make_env(samples_per_iteration=20, max_conversions_per_iteration=2)
            method = make_method(env, seed=9, perturbation_scale=scale)
            return run_baseline(env, method, 10, seed=9).team_rewards

        assert not np.allclose(run(0.0), run(0.9))

    def test_a_proposal_pool_smaller_than_the_demand_proposes_everything(self) -> None:
        env = make_env(samples_per_iteration=6, max_conversions_per_iteration=4)
        method = make_method(env)
        batch = env.reset(seed=11)
        method.reset(env, 11)
        decision = method.decide(batch)
        assert set(decision.proposals or ()) == set(batch.parcels)
