"""Tests for the QOLU reward composition.

R(s, a) = sum_i w_i R_phi_i(s, a) + R_society(s, a)   (Section 3.1 of the paper)
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.config import SCENARIOS, ScenarioWeights
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.env.rewards import (
    RewardConfig,
    TopDownObjective,
    compose_team_reward,
    land_use_separation_reward,
    objective_rewards,
    societal_reward,
    urban_proximity_reward,
    woodland_preservation_reward,
)
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index


def uniform_grid(land_class: AggregateClass, size: int = 5) -> ParcelGrid:
    composition = np.zeros((N_AGGREGATE_CLASSES, size, size), dtype=np.float32)
    composition[band_index(land_class)] = 1.0
    return ParcelGrid(composition)


def grid_with(
    features: dict[tuple[int, int], AggregateClass], size: int = 5
) -> ParcelGrid:
    composition = np.zeros((N_AGGREGATE_CLASSES, size, size), dtype=np.float32)
    composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 1.0
    for (row, col), land_class in features.items():
        composition[:, row, col] = 0.0
        composition[band_index(land_class), row, col] = 1.0
    return ParcelGrid(composition)


class TestScenarioWeights:
    def test_the_four_published_scenarios_are_available(self) -> None:
        assert set(SCENARIOS) == {1, 2, 3, 4}
        assert SCENARIOS[1] == ScenarioWeights(land_use=0.33, woodland=0.33, urban=0.33)
        assert SCENARIOS[2] == ScenarioWeights(land_use=0.50, woodland=0.25, urban=0.25)
        assert SCENARIOS[3] == ScenarioWeights(land_use=0.25, woodland=0.50, urban=0.25)
        assert SCENARIOS[4] == ScenarioWeights(land_use=0.25, woodland=0.25, urban=0.50)

    def test_weights_expose_an_objective_lookup(self) -> None:
        weights = SCENARIOS[3]
        assert weights[TopDownObjective.LAND_USE_SEPARATION] == pytest.approx(0.25)
        assert weights[TopDownObjective.WOODLAND_PRESERVATION] == pytest.approx(0.50)
        assert weights[TopDownObjective.URBAN_PROXIMITY] == pytest.approx(0.25)

    def test_rejects_negative_weights(self) -> None:
        with pytest.raises(ValueError):
            ScenarioWeights(land_use=-0.1, woodland=0.5, urban=0.6)

    def test_can_be_normalised(self) -> None:
        normalised = ScenarioWeights(
            land_use=0.33, woodland=0.33, urban=0.33
        ).normalised()
        assert normalised.total() == pytest.approx(1.0)
        assert normalised.land_use == pytest.approx(1 / 3)


class TestWoodlandPreservationReward:
    def test_perfect_score_when_no_protected_cover_is_lost(self) -> None:
        grid = uniform_grid(AggregateClass.SEMI_NATURAL_GRASSLAND)
        _, conversions = grid.convert_to_urban([ParcelIndex(2, 2)], fraction=1.0)
        reward = woodland_preservation_reward(grid, conversions, RewardConfig())
        assert reward == pytest.approx(1.0)

    def test_worst_score_when_a_fully_wooded_parcel_is_lost(self) -> None:
        grid = uniform_grid(AggregateClass.BROADLEAF_WOODLAND)
        _, conversions = grid.convert_to_urban([ParcelIndex(2, 2)], fraction=1.0)
        reward = woodland_preservation_reward(grid, conversions, RewardConfig())
        assert reward == pytest.approx(0.0)

    def test_scales_with_the_fraction_of_protected_cover_lost(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 3, 3), dtype=np.float32)
        composition[band_index(AggregateClass.CONIFEROUS_WOODLAND)] = 0.25
        composition[band_index(AggregateClass.ARABLE)] = 0.25
        composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 0.5
        grid = ParcelGrid(composition)
        _, conversions = grid.convert_to_urban([ParcelIndex(1, 1)], fraction=1.0)
        assert woodland_preservation_reward(
            grid, conversions, RewardConfig()
        ) == pytest.approx(0.5)

    def test_partial_conversion_loses_proportionally_less(self) -> None:
        grid = uniform_grid(AggregateClass.BROADLEAF_WOODLAND)
        _, conversions = grid.convert_to_urban([ParcelIndex(0, 0)], fraction=0.5)
        assert woodland_preservation_reward(
            grid, conversions, RewardConfig()
        ) == pytest.approx(0.5)

    def test_averages_over_all_conversions(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 2, 2), dtype=np.float32)
        composition[band_index(AggregateClass.BROADLEAF_WOODLAND), 0, 0] = 1.0
        composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND), 0, 1] = 1.0
        grid = ParcelGrid(composition)
        _, conversions = grid.convert_to_urban(
            [ParcelIndex(0, 0), ParcelIndex(0, 1)], fraction=1.0
        )
        assert woodland_preservation_reward(
            grid, conversions, RewardConfig()
        ) == pytest.approx(0.5)

    def test_no_conversions_is_neutral(self) -> None:
        grid = uniform_grid(AggregateClass.BROADLEAF_WOODLAND)
        assert woodland_preservation_reward(grid, (), RewardConfig()) == pytest.approx(
            0.0
        )


class TestUrbanProximityReward:
    def test_developing_next_to_an_existing_town_scores_higher(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.BUILT_UP})
        _, near = grid.convert_to_urban([ParcelIndex(0, 1)], fraction=1.0)
        _, far = grid.convert_to_urban([ParcelIndex(4, 4)], fraction=1.0)
        config = RewardConfig()
        assert urban_proximity_reward(grid, near, config) > urban_proximity_reward(
            grid, far, config
        )

    def test_score_is_bounded_in_the_unit_interval(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.BUILT_UP})
        _, conversions = grid.convert_to_urban([ParcelIndex(2, 2)], fraction=1.0)
        reward = urban_proximity_reward(grid, conversions, RewardConfig())
        assert 0.0 <= reward <= 1.0

    def test_uses_the_pre_conversion_urban_mask(self) -> None:
        # Without an existing urban parcel the reward must be 0, even though the
        # converted parcel itself becomes urban.
        grid = grid_with({})
        _, conversions = grid.convert_to_urban([ParcelIndex(2, 2)], fraction=1.0)
        assert urban_proximity_reward(
            grid, conversions, RewardConfig()
        ) == pytest.approx(0.0)

    def test_adjacent_development_scores_close_to_one(self) -> None:
        grid = grid_with({(2, 2): AggregateClass.BUILT_UP}, size=41)
        _, conversions = grid.convert_to_urban([ParcelIndex(2, 3)], fraction=1.0)
        assert urban_proximity_reward(grid, conversions, RewardConfig()) > 0.95


class TestLandUseSeparationReward:
    def test_developing_away_from_protected_habitat_scores_higher(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.MOUNTAIN_HEATH_BOG})
        _, near = grid.convert_to_urban([ParcelIndex(0, 1)], fraction=1.0)
        _, far = grid.convert_to_urban([ParcelIndex(4, 4)], fraction=1.0)
        config = RewardConfig()
        assert land_use_separation_reward(
            grid, far, config
        ) > land_use_separation_reward(grid, near, config)

    def test_absent_habitat_gives_the_maximum_score(self) -> None:
        grid = grid_with({})
        _, conversions = grid.convert_to_urban([ParcelIndex(1, 1)], fraction=1.0)
        assert land_use_separation_reward(
            grid, conversions, RewardConfig()
        ) == pytest.approx(1.0)

    def test_score_is_bounded_in_the_unit_interval(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.MOUNTAIN_HEATH_BOG})
        _, conversions = grid.convert_to_urban([ParcelIndex(3, 3)], fraction=1.0)
        reward = land_use_separation_reward(grid, conversions, RewardConfig())
        assert 0.0 <= reward <= 1.0


class TestSocietalReward:
    def test_unchanged_landscape_scores_alpha_plus_beta(self) -> None:
        grid = grid_with({})
        config = RewardConfig(alpha=0.5, beta=0.5)
        assert societal_reward(grid, grid, config) == pytest.approx(1.0)

    def test_losing_half_the_forest_halves_the_forest_term(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 1, 2), dtype=np.float32)
        composition[band_index(AggregateClass.BROADLEAF_WOODLAND)] = 1.0
        initial = ParcelGrid(composition)
        current, _ = initial.convert_to_urban([ParcelIndex(0, 0)], fraction=1.0)
        config = RewardConfig(alpha=1.0, beta=0.0)
        assert societal_reward(initial, current, config) == pytest.approx(0.5)

    def test_alpha_and_beta_weight_forest_and_agricultural_terms(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 1, 2), dtype=np.float32)
        composition[band_index(AggregateClass.BROADLEAF_WOODLAND), 0, 0] = 1.0
        composition[band_index(AggregateClass.ARABLE), 0, 1] = 1.0
        initial = ParcelGrid(composition)
        current, _ = initial.convert_to_urban([ParcelIndex(0, 1)], fraction=1.0)
        config = RewardConfig(alpha=0.25, beta=0.75)
        # forest fully intact (1.0), agricultural fully lost (0.0)
        assert societal_reward(initial, current, config) == pytest.approx(0.25)

    def test_absent_land_types_count_as_fully_preserved(self) -> None:
        grid = grid_with({})  # no forest, no agricultural land
        config = RewardConfig(alpha=0.5, beta=0.5)
        assert societal_reward(grid, grid, config) == pytest.approx(1.0)


class TestTeamRewardComposition:
    def test_matches_the_paper_formula(self) -> None:
        rewards = {
            TopDownObjective.LAND_USE_SEPARATION: 0.2,
            TopDownObjective.WOODLAND_PRESERVATION: 0.4,
            TopDownObjective.URBAN_PROXIMITY: 0.6,
        }
        weights = ScenarioWeights(land_use=0.5, woodland=0.25, urban=0.25)
        team = compose_team_reward(rewards, weights, societal=0.8)
        expected = 0.5 * 0.2 + 0.25 * 0.4 + 0.25 * 0.6 + 0.8
        assert team == pytest.approx(expected)

    def test_is_linear_in_the_objective_rewards(self) -> None:
        weights = SCENARIOS[1]
        base = {objective: 0.5 for objective in TopDownObjective}
        doubled = {objective: 1.0 for objective in TopDownObjective}
        team_base = compose_team_reward(base, weights, societal=0.0)
        team_doubled = compose_team_reward(doubled, weights, societal=0.0)
        assert team_doubled == pytest.approx(2.0 * team_base)

    def test_scenario_weighting_changes_the_emphasis(self) -> None:
        rewards = {
            TopDownObjective.LAND_USE_SEPARATION: 1.0,
            TopDownObjective.WOODLAND_PRESERVATION: 0.0,
            TopDownObjective.URBAN_PROXIMITY: 0.0,
        }
        land_focus = compose_team_reward(rewards, SCENARIOS[2], societal=0.0)
        woodland_focus = compose_team_reward(rewards, SCENARIOS[3], societal=0.0)
        assert land_focus > woodland_focus

    def test_requires_every_objective(self) -> None:
        with pytest.raises(KeyError):
            compose_team_reward(
                {TopDownObjective.LAND_USE_SEPARATION: 1.0}, SCENARIOS[1], societal=0.0
            )


class TestObjectiveRewardBundle:
    def test_returns_one_entry_per_top_down_objective(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.BUILT_UP})
        after, conversions = grid.convert_to_urban([ParcelIndex(1, 1)], fraction=1.0)
        rewards = objective_rewards(grid, after, conversions, RewardConfig())
        assert set(rewards) == set(TopDownObjective)
        assert all(0.0 <= value <= 1.0 for value in rewards.values())
