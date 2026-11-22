"""Tests for per-parcel objective compliance and the QOLU parcel rewards.

Section 3.5: "Each QOLU agent then allocates rewards based on the degree of
compliance with their individual objectives: R_QOLU,i = f(parcel_data,
objectives_i)".
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.config import QOLURewardConfig, RewardConfig
from chapter4_qolu.env.compliance import compliance_scores, parcel_compliance
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.env.qolu_rewards import qolu_parcel_rewards
from chapter4_qolu.env.urban_env import APPROVE, REJECT
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index
from chapter4_qolu.objectives import TopDownObjective


def grid_with(
    features: dict[tuple[int, int], AggregateClass], size: int = 5
) -> ParcelGrid:
    composition = np.zeros((N_AGGREGATE_CLASSES, size, size), dtype=np.float32)
    composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 1.0
    for (row, col), land_class in features.items():
        composition[:, row, col] = 0.0
        composition[band_index(land_class), row, col] = 1.0
    return ParcelGrid(composition)


class TestLandUseSeparationCompliance:
    def test_parcels_far_from_habitat_comply_better(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.MOUNTAIN_HEATH_BOG})
        config = RewardConfig()
        near = parcel_compliance(
            grid, ParcelIndex(0, 1), TopDownObjective.LAND_USE_SEPARATION, config
        )
        far = parcel_compliance(
            grid, ParcelIndex(4, 4), TopDownObjective.LAND_USE_SEPARATION, config
        )
        assert far > near

    def test_building_on_the_habitat_scores_zero(self) -> None:
        grid = grid_with({(2, 2): AggregateClass.MOUNTAIN_HEATH_BOG})
        assert parcel_compliance(
            grid,
            ParcelIndex(2, 2),
            TopDownObjective.LAND_USE_SEPARATION,
            RewardConfig(),
        ) == pytest.approx(0.0)

    def test_absent_habitat_scores_one(self) -> None:
        grid = grid_with({})
        assert parcel_compliance(
            grid,
            ParcelIndex(2, 2),
            TopDownObjective.LAND_USE_SEPARATION,
            RewardConfig(),
        ) == pytest.approx(1.0)


class TestWoodlandPreservationCompliance:
    def test_grassland_parcel_destroys_nothing_protected(self) -> None:
        grid = grid_with({})
        assert parcel_compliance(
            grid,
            ParcelIndex(1, 1),
            TopDownObjective.WOODLAND_PRESERVATION,
            RewardConfig(),
        ) == pytest.approx(1.0)

    def test_fully_wooded_parcel_scores_zero(self) -> None:
        grid = grid_with({(1, 1): AggregateClass.BROADLEAF_WOODLAND})
        assert parcel_compliance(
            grid,
            ParcelIndex(1, 1),
            TopDownObjective.WOODLAND_PRESERVATION,
            RewardConfig(),
        ) == pytest.approx(0.0)

    def test_scales_with_the_protected_share_of_the_parcel(self) -> None:
        composition = np.zeros((N_AGGREGATE_CLASSES, 2, 2), dtype=np.float32)
        composition[band_index(AggregateClass.ARABLE), 0, 0] = 0.25
        composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND), 0, 0] = 0.75
        grid = ParcelGrid(composition)
        assert parcel_compliance(
            grid,
            ParcelIndex(0, 0),
            TopDownObjective.WOODLAND_PRESERVATION,
            RewardConfig(),
        ) == pytest.approx(0.75)


class TestUrbanProximityCompliance:
    def test_parcels_next_to_town_comply_better(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.BUILT_UP})
        config = RewardConfig()
        near = parcel_compliance(
            grid, ParcelIndex(0, 1), TopDownObjective.URBAN_PROXIMITY, config
        )
        far = parcel_compliance(
            grid, ParcelIndex(4, 4), TopDownObjective.URBAN_PROXIMITY, config
        )
        assert near > far

    def test_no_urban_land_scores_zero(self) -> None:
        grid = grid_with({})
        assert parcel_compliance(
            grid, ParcelIndex(2, 2), TopDownObjective.URBAN_PROXIMITY, RewardConfig()
        ) == pytest.approx(0.0)


class TestComplianceScores:
    def test_returns_one_score_per_parcel_in_order(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.BUILT_UP})
        parcels = (ParcelIndex(0, 1), ParcelIndex(4, 4), ParcelIndex(2, 2))
        scores = compliance_scores(
            grid, parcels, TopDownObjective.URBAN_PROXIMITY, RewardConfig()
        )
        assert scores.shape == (3,)
        expected = [
            parcel_compliance(
                grid, parcel, TopDownObjective.URBAN_PROXIMITY, RewardConfig()
            )
            for parcel in parcels
        ]
        np.testing.assert_allclose(scores, expected)

    def test_every_objective_is_bounded_in_the_unit_interval(self) -> None:
        grid = grid_with(
            {
                (0, 0): AggregateClass.BUILT_UP,
                (4, 4): AggregateClass.MOUNTAIN_HEATH_BOG,
                (2, 2): AggregateClass.BROADLEAF_WOODLAND,
            }
        )
        parcels = tuple(ParcelIndex(row, col) for row in range(5) for col in range(5))
        for objective in TopDownObjective:
            scores = compliance_scores(grid, parcels, objective, RewardConfig())
            assert float(scores.min()) >= 0.0
            assert float(scores.max()) <= 1.0

    def test_empty_parcel_list_gives_an_empty_result(self) -> None:
        scores = compliance_scores(
            grid_with({}), (), TopDownObjective.URBAN_PROXIMITY, RewardConfig()
        )
        assert scores.shape == (0,)


class TestQOLUParcelRewards:
    def test_symmetric_form_rewards_approving_compliant_parcels(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.BROADLEAF_WOODLAND})
        parcels = (ParcelIndex(0, 0), ParcelIndex(2, 2))  # woodland, grassland
        actions = np.array([APPROVE, APPROVE])
        rewards = qolu_parcel_rewards(
            grid,
            parcels,
            actions,
            TopDownObjective.WOODLAND_PRESERVATION,
            proposed=None,
            config=RewardConfig(),
            qolu_reward=QOLURewardConfig(),
            team_reward=0.0,
        )
        # compliance 0.0 (destroys woodland) and 1.0 (destroys nothing)
        np.testing.assert_allclose(rewards, [-0.5, 0.5])

    def test_symmetric_form_rewards_rejecting_non_compliant_parcels(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.BROADLEAF_WOODLAND})
        parcels = (ParcelIndex(0, 0), ParcelIndex(2, 2))
        actions = np.array([REJECT, REJECT])
        rewards = qolu_parcel_rewards(
            grid,
            parcels,
            actions,
            TopDownObjective.WOODLAND_PRESERVATION,
            proposed=None,
            config=RewardConfig(),
            qolu_reward=QOLURewardConfig(),
            team_reward=0.0,
        )
        np.testing.assert_allclose(rewards, [0.5, -0.5])

    def test_asymmetric_form_gives_no_signal_for_rejection(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.BROADLEAF_WOODLAND})
        parcels = (ParcelIndex(0, 0), ParcelIndex(2, 2))
        actions = np.array([REJECT, APPROVE])
        rewards = qolu_parcel_rewards(
            grid,
            parcels,
            actions,
            TopDownObjective.WOODLAND_PRESERVATION,
            proposed=None,
            config=RewardConfig(),
            qolu_reward=QOLURewardConfig(approval_form="asymmetric"),
            team_reward=0.0,
        )
        np.testing.assert_allclose(rewards, [0.0, 0.5])

    def test_team_reward_can_be_mixed_in(self) -> None:
        grid = grid_with({})
        parcels = (ParcelIndex(1, 1),)
        rewards = qolu_parcel_rewards(
            grid,
            parcels,
            np.array([APPROVE]),
            TopDownObjective.WOODLAND_PRESERVATION,
            proposed=None,
            config=RewardConfig(),
            qolu_reward=QOLURewardConfig(team_reward_weight=2.0),
            team_reward=1.5,
        )
        np.testing.assert_allclose(rewards, [0.5 + 2.0 * 1.5])

    def test_team_reward_is_not_mixed_in_by_default(self) -> None:
        assert QOLURewardConfig().team_reward_weight == pytest.approx(0.0)

    def test_unproposed_parcels_can_be_down_weighted(self) -> None:
        grid = grid_with({})
        parcels = (ParcelIndex(1, 1), ParcelIndex(2, 2))
        rewards = qolu_parcel_rewards(
            grid,
            parcels,
            np.array([APPROVE, APPROVE]),
            TopDownObjective.WOODLAND_PRESERVATION,
            proposed={ParcelIndex(1, 1)},
            config=RewardConfig(),
            qolu_reward=QOLURewardConfig(unproposed_reward_scale=0.0),
            team_reward=0.0,
        )
        np.testing.assert_allclose(rewards, [0.5, 0.0])

    def test_appraisals_count_even_when_the_conversion_did_not_happen(self) -> None:
        grid = grid_with({})
        parcels = (ParcelIndex(1, 1),)
        rewards = qolu_parcel_rewards(
            grid,
            parcels,
            np.array([APPROVE]),
            TopDownObjective.WOODLAND_PRESERVATION,
            proposed=set(),  # BUIA proposed nothing, so nothing was converted
            config=RewardConfig(),
            qolu_reward=QOLURewardConfig(),
            team_reward=0.0,
        )
        np.testing.assert_allclose(rewards, [0.5])

    def test_rewards_depend_on_both_the_parcel_and_the_action(self) -> None:
        grid = grid_with({(0, 0): AggregateClass.BROADLEAF_WOODLAND})
        parcels = (ParcelIndex(0, 0), ParcelIndex(0, 0))
        rewards = qolu_parcel_rewards(
            grid,
            parcels,
            np.array([APPROVE, REJECT]),
            TopDownObjective.WOODLAND_PRESERVATION,
            proposed=None,
            config=RewardConfig(),
            qolu_reward=QOLURewardConfig(),
            team_reward=0.0,
        )
        assert rewards[0] != rewards[1]

    def test_rejects_mismatched_action_count(self) -> None:
        with pytest.raises(ValueError):
            qolu_parcel_rewards(
                grid_with({}),
                (ParcelIndex(0, 0),),
                np.array([APPROVE, REJECT]),
                TopDownObjective.WOODLAND_PRESERVATION,
                proposed=None,
                config=RewardConfig(),
                qolu_reward=QOLURewardConfig(),
                team_reward=0.0,
            )

    def test_rejects_an_unknown_approval_form(self) -> None:
        with pytest.raises(ValueError):
            QOLURewardConfig(approval_form="sigmoid")

    def test_rejects_an_unknown_mode(self) -> None:
        with pytest.raises(ValueError):
            QOLURewardConfig(mode="global")

    def test_per_parcel_is_the_default_mode(self) -> None:
        assert QOLURewardConfig().mode == "per_parcel"
        assert QOLURewardConfig().approval_form == "symmetric"
        assert QOLURewardConfig().unproposed_reward_scale == pytest.approx(1.0)
