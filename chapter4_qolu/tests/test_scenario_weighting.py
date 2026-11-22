"""Tests for the two optional routes by which the scenario weights reach the
top-down agents.

The paper makes the scenario weights *per-stakeholder preference* weights: the
shared reward is ``R = sum_i w_i R_phi_i + R_society`` (Section 3.1) and the
land-preference metric is ``LP = sum_i w_i P_phi_TD_i`` (Eq. 3).  In the default
``per_parcel`` reading neither of those reaches a top-down agent - stakeholder
``i`` is paid only for compliance with its *own* objective and the consensus is
a plain sum of quality scores - so QOLU + BUIA (and P-MADDPG, which shares both
code paths) produces byte-identical trajectories for Scenarios 1, 2 and 3.

Two optional switches close that gap:

* ``QOLURewardConfig.scale_by_scenario_weight`` - the per-parcel reading of
  ``sum_i w_i R_phi_i``;
* ``QOLURewardConfig.weighted_consensus`` - the weights as the instrument that
  "balances ecological, economic, and policy-related objectives".

Both are off by default, and both have to reproduce the unweighted behaviour
exactly when the weights are equal.
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.agent.consensus import (
    action_weighted_votes,
    balanced_weights,
    consensus_decisions,
)
from chapter4_qolu.config import (
    SCENARIOS,
    QOLURewardConfig,
    RewardConfig,
    ScenarioWeights,
)
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.env.qolu_rewards import (
    qolu_parcel_rewards,
    scenario_weight_multiplier,
)
from chapter4_qolu.env.urban_env import APPROVE, REJECT
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index
from chapter4_qolu.objectives import TopDownObjective

#: The three objectives in declaration order, i.e. the order both
#: ``runner._stakeholder_objectives`` and ``PMADDPGMethod`` assign them in.
OBJECTIVES: tuple[TopDownObjective, ...] = tuple(TopDownObjective)

#: Proximity saturates at 5 km on the synthetic landscape below, so a parcel
#: eight cells from the bog block is "far" rather than "0.6 of a diagonal".
SATURATING = RewardConfig(max_distance=5.0)


def woodland_landscape(size: int = 10) -> ParcelGrid:
    """Grassland, an urban corner, a woodland block beside it, a distant bog.

    Converting a parcel of the woodland block is *good* for the land-use and
    urban-proximity stakeholders (far from the bog, next to the town) and
    maximally *bad* for the woodland guardian, which is exactly the conflict the
    scenario weights are supposed to arbitrate.
    """
    composition = np.zeros((N_AGGREGATE_CLASSES, size, size), dtype=np.float32)
    composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 1.0
    for row, col in ((0, 0), (0, 1), (1, 0)):
        composition[:, row, col] = 0.0
        composition[band_index(AggregateClass.BUILT_UP), row, col] = 1.0
    composition[:, 0:3, 2:5] = 0.0
    composition[band_index(AggregateClass.BROADLEAF_WOODLAND), 0:3, 2:5] = 1.0
    composition[:, size - 3 :, size - 3 :] = 0.0
    composition[
        band_index(AggregateClass.MOUNTAIN_HEATH_BOG), size - 3 :, size - 3 :
    ] = 1.0
    return ParcelGrid(composition)


def decision_buffer() -> tuple[ParcelIndex, ...]:
    """Nine woodland parcels plus four parcels that destroy no woodland."""
    woodland = [ParcelIndex(row, col) for row in range(3) for col in range(2, 5)]
    other = [ParcelIndex(1, 1), ParcelIndex(2, 1), ParcelIndex(3, 3), ParcelIndex(4, 4)]
    return tuple(woodland + other)


def converged_quality_scores(
    grid: ParcelGrid,
    parcels: tuple[ParcelIndex, ...],
    weights: ScenarioWeights,
    qolu_reward: QOLURewardConfig,
    n_stakeholders: int = 3,
) -> list[np.ndarray]:
    """What a stakeholder that has converged on its per-parcel reward scores.

    A greedy QOLU agent ranks an appraisal by ``Q(approve) - Q(reject)``, whose
    fixed point under the per-parcel signal is the reward gap between the two
    actions.  Computing that gap from :func:`qolu_parcel_rewards` itself keeps
    the test on the code path under test rather than on a hand-written formula.
    """
    scores: list[np.ndarray] = []
    for objective in OBJECTIVES:
        gap = [
            qolu_parcel_rewards(
                grid,
                parcels,
                np.full(len(parcels), action, dtype=np.int64),
                objective,
                proposed=None,
                config=SATURATING,
                qolu_reward=qolu_reward,
                team_reward=0.0,
                scenario_weights=weights,
                n_stakeholders=n_stakeholders,
            )
            for action in (APPROVE, REJECT)
        ]
        scores.append(gap[0] - gap[1])
    return scores


def greedy_actions(scores: list[np.ndarray]) -> list[np.ndarray]:
    return [(score > 0.0).astype(np.int64) for score in scores]


class TestDefaults:
    """The default reading is the unweighted one: both switches start off."""

    def test_both_switches_are_off_by_default(self) -> None:
        config = QOLURewardConfig()
        assert config.scale_by_scenario_weight is False
        assert config.weighted_consensus is False

    def test_the_default_reward_ignores_the_weights_entirely(self) -> None:
        grid = woodland_landscape()
        parcels = (ParcelIndex(0, 2),)
        rewards = [
            qolu_parcel_rewards(
                grid,
                parcels,
                np.array([APPROVE]),
                TopDownObjective.WOODLAND_PRESERVATION,
                proposed=None,
                config=SATURATING,
                qolu_reward=QOLURewardConfig(),
                team_reward=0.0,
                scenario_weights=SCENARIOS[scenario],
                n_stakeholders=3,
            )
            for scenario in (1, 3)
        ]
        np.testing.assert_array_equal(rewards[0], rewards[1])

    def test_the_weights_may_be_omitted_while_the_switch_is_off(self) -> None:
        rewards = qolu_parcel_rewards(
            woodland_landscape(),
            (ParcelIndex(4, 4),),
            np.array([APPROVE]),
            TopDownObjective.WOODLAND_PRESERVATION,
            proposed=None,
            config=SATURATING,
            qolu_reward=QOLURewardConfig(),
            team_reward=0.0,
        )
        assert rewards.shape == (1,)


class TestScenarioWeightMultiplier:
    """``n_stakeholders * w_i``, with the weighting normalised to sum to one."""

    def test_equal_weights_give_exactly_one(self) -> None:
        for objective in OBJECTIVES:
            multiplier = scenario_weight_multiplier(SCENARIOS[1], objective, 3)
            assert multiplier == 1.0  # exactly, not approximately

    def test_scenario_three_amplifies_the_woodland_guardian(self) -> None:
        multipliers = [
            scenario_weight_multiplier(SCENARIOS[3], objective, 3)
            for objective in OBJECTIVES
        ]
        assert multipliers == pytest.approx([0.75, 1.5, 0.75])

    def test_scenario_two_amplifies_the_land_use_stakeholder(self) -> None:
        multipliers = [
            scenario_weight_multiplier(SCENARIOS[2], objective, 3)
            for objective in OBJECTIVES
        ]
        assert multipliers == pytest.approx([1.5, 0.75, 0.75])

    def test_every_scenario_averages_to_one(self) -> None:
        for scenario in sorted(SCENARIOS):
            multipliers = [
                scenario_weight_multiplier(SCENARIOS[scenario], objective, 3)
                for objective in OBJECTIVES
            ]
            assert float(np.mean(multipliers)) == pytest.approx(1.0)

    def test_rejects_a_non_positive_stakeholder_count(self) -> None:
        with pytest.raises(ValueError):
            scenario_weight_multiplier(SCENARIOS[1], OBJECTIVES[0], 0)

    def test_rejects_an_all_zero_weighting(self) -> None:
        with pytest.raises(ValueError):
            scenario_weight_multiplier(
                ScenarioWeights(land_use=0.0, woodland=0.0, urban=0.0), OBJECTIVES[0], 3
            )


class TestRewardScaling:
    def _reward(
        self, objective: TopDownObjective, scenario: int, scale: bool, team: float = 0.0
    ) -> float:
        rewards = qolu_parcel_rewards(
            woodland_landscape(),
            (ParcelIndex(4, 4),),
            np.array([APPROVE]),
            objective,
            proposed=None,
            config=SATURATING,
            qolu_reward=QOLURewardConfig(
                scale_by_scenario_weight=scale, team_reward_weight=team
            ),
            team_reward=1.0,
            scenario_weights=SCENARIOS[scenario],
            n_stakeholders=3,
        )
        return float(rewards[0])

    def test_equal_weights_reproduce_the_unscaled_reward_exactly(self) -> None:
        for objective in OBJECTIVES:
            assert self._reward(objective, 1, scale=True) == self._reward(
                objective, 1, scale=False
            )

    def test_scenario_three_amplifies_the_woodland_guardian(self) -> None:
        plain = self._reward(TopDownObjective.WOODLAND_PRESERVATION, 3, scale=False)
        scaled = self._reward(TopDownObjective.WOODLAND_PRESERVATION, 3, scale=True)
        assert scaled == pytest.approx(1.5 * plain)

    def test_scenario_three_damps_the_other_two_stakeholders(self) -> None:
        for objective in (
            TopDownObjective.LAND_USE_SEPARATION,
            TopDownObjective.URBAN_PROXIMITY,
        ):
            plain = self._reward(objective, 3, scale=False)
            assert self._reward(objective, 3, scale=True) == pytest.approx(0.75 * plain)

    def test_the_shared_team_term_is_not_scaled(self) -> None:
        objective = TopDownObjective.WOODLAND_PRESERVATION
        own = self._reward(objective, 3, scale=True)
        mixed = self._reward(objective, 3, scale=True, team=2.0)
        # R = w_i * R_phi_i + alpha * team_reward: only the first term is weighted.
        assert mixed - own == pytest.approx(2.0)

    def test_requires_the_weights_when_the_switch_is_on(self) -> None:
        with pytest.raises(ValueError, match="scale_by_scenario_weight"):
            qolu_parcel_rewards(
                woodland_landscape(),
                (ParcelIndex(4, 4),),
                np.array([APPROVE]),
                TopDownObjective.WOODLAND_PRESERVATION,
                proposed=None,
                config=SATURATING,
                qolu_reward=QOLURewardConfig(scale_by_scenario_weight=True),
                team_reward=0.0,
            )

    def test_requires_the_stakeholder_count_when_the_switch_is_on(self) -> None:
        with pytest.raises(ValueError, match="scale_by_scenario_weight"):
            qolu_parcel_rewards(
                woodland_landscape(),
                (ParcelIndex(4, 4),),
                np.array([APPROVE]),
                TopDownObjective.WOODLAND_PRESERVATION,
                proposed=None,
                config=SATURATING,
                qolu_reward=QOLURewardConfig(scale_by_scenario_weight=True),
                team_reward=0.0,
                scenario_weights=SCENARIOS[3],
            )

    def test_an_empty_buffer_still_returns_an_empty_result(self) -> None:
        rewards = qolu_parcel_rewards(
            woodland_landscape(),
            (),
            np.zeros(0, dtype=np.int64),
            TopDownObjective.WOODLAND_PRESERVATION,
            proposed=None,
            config=SATURATING,
            qolu_reward=QOLURewardConfig(scale_by_scenario_weight=True),
            team_reward=0.0,
            scenario_weights=SCENARIOS[3],
            n_stakeholders=3,
        )
        assert rewards.shape == (0,)


class TestBalancedWeights:
    def test_equal_weights_become_exactly_one(self) -> None:
        np.testing.assert_array_equal(
            balanced_weights([0.33, 0.33, 0.33]), np.ones(3, dtype=np.float64)
        )

    def test_scenario_three_weights_average_to_one(self) -> None:
        values = balanced_weights([0.25, 0.50, 0.25])
        np.testing.assert_allclose(values, [0.75, 1.5, 0.75])
        assert float(values.mean()) == pytest.approx(1.0)

    def test_a_single_stakeholder_is_never_rescaled(self) -> None:
        np.testing.assert_array_equal(balanced_weights([0.25]), np.ones(1))

    def test_rejects_an_empty_weighting(self) -> None:
        with pytest.raises(ValueError):
            balanced_weights([])

    def test_rejects_a_negative_weight(self) -> None:
        with pytest.raises(ValueError):
            balanced_weights([0.5, -0.5, 0.5])

    def test_rejects_an_all_zero_weighting(self) -> None:
        with pytest.raises(ValueError):
            balanced_weights([0.0, 0.0, 0.0])


def sample_scores() -> list[np.ndarray]:
    """Three stakeholders scoring two candidates; index 1 is the guardian."""
    return [np.array([1.0, -2.0]), np.array([0.5, 0.5]), np.array([-0.25, 1.0])]


class TestWeightedVotes:
    def test_omitting_the_weights_is_the_published_rule(self) -> None:
        scores = sample_scores()
        votes = action_weighted_votes(scores, greedy_actions(scores))
        np.testing.assert_array_equal(
            consensus_decisions(votes), consensus_decisions(scores)
        )

    def test_equal_weights_reproduce_the_unweighted_votes_exactly(self) -> None:
        scores = sample_scores()
        actions = greedy_actions(scores)
        plain = action_weighted_votes(scores, actions)
        weighted = action_weighted_votes(scores, actions, weights=[0.33, 0.33, 0.33])
        for expected, actual in zip(plain, weighted):
            np.testing.assert_array_equal(expected, actual)

    def test_scenario_three_amplifies_the_woodland_vote(self) -> None:
        scores = sample_scores()
        actions = greedy_actions(scores)
        weighted = action_weighted_votes(scores, actions, weights=[0.25, 0.50, 0.25])
        plain = action_weighted_votes(scores, actions)
        np.testing.assert_allclose(weighted[1], 1.5 * plain[1])
        np.testing.assert_allclose(weighted[0], 0.75 * plain[0])

    def test_weighting_keeps_the_sign_of_the_action_taken(self) -> None:
        votes = action_weighted_votes(
            [np.array([2.0])], [np.array([REJECT])], weights=[0.5]
        )
        assert votes[0][0] < 0.0

    def test_rejects_a_mismatched_weight_count(self) -> None:
        scores = sample_scores()
        with pytest.raises(ValueError):
            action_weighted_votes(scores, greedy_actions(scores), weights=[1.0, 1.0])


class TestWoodlandApprovalsOnASyntheticBuffer:
    """Scenario 3 has to restrain development on woodland.

    The buffer holds nine woodland parcels beside the town - which the land-use
    and urban-proximity stakeholders both want - and four parcels that destroy
    no woodland.  Under the published plain sum the woodland guardian is
    outvoted; each switch gives it the weight Table 3 assigns it in Scenario 3.
    """

    @staticmethod
    def _approvals_via_rewards(scenario: int, scale: bool) -> int:
        grid = woodland_landscape()
        parcels = decision_buffer()
        scores = converged_quality_scores(
            grid,
            parcels,
            SCENARIOS[scenario],
            QOLURewardConfig(scale_by_scenario_weight=scale),
        )
        return int(consensus_decisions(scores).sum())

    @staticmethod
    def _approvals_via_consensus(scenario: int, weighted: bool) -> int:
        grid = woodland_landscape()
        parcels = decision_buffer()
        scores = converged_quality_scores(
            grid, parcels, SCENARIOS[scenario], QOLURewardConfig()
        )
        weights = (
            [SCENARIOS[scenario][objective] for objective in OBJECTIVES]
            if weighted
            else None
        )
        votes = action_weighted_votes(scores, greedy_actions(scores), weights=weights)
        return int(consensus_decisions(votes).sum())

    def test_today_scenario_one_and_three_approve_the_same_parcels(self) -> None:
        # The invariance the two switches make optional, kept as a regression
        # test: with both switches off the weights change nothing.
        assert self._approvals_via_rewards(1, scale=False) == (
            self._approvals_via_rewards(3, scale=False)
        )
        assert self._approvals_via_consensus(1, weighted=False) == (
            self._approvals_via_consensus(3, weighted=False)
        )

    def test_reward_scaling_approves_fewer_parcels_under_scenario_three(self) -> None:
        under_one = self._approvals_via_rewards(1, scale=True)
        under_three = self._approvals_via_rewards(3, scale=True)
        assert under_one > 0
        assert under_three < under_one

    def test_weighted_consensus_approves_fewer_parcels_under_scenario_three(
        self,
    ) -> None:
        under_one = self._approvals_via_consensus(1, weighted=True)
        under_three = self._approvals_via_consensus(3, weighted=True)
        assert under_one > 0
        assert under_three < under_one

    def test_the_parcels_that_flip_are_the_woodland_ones(self) -> None:
        grid = woodland_landscape()
        parcels = decision_buffer()
        plain = consensus_decisions(
            converged_quality_scores(grid, parcels, SCENARIOS[1], QOLURewardConfig())
        )
        scaled = consensus_decisions(
            converged_quality_scores(
                grid,
                parcels,
                SCENARIOS[3],
                QOLURewardConfig(scale_by_scenario_weight=True),
            )
        )
        flipped = {
            parcels[index]
            for index in range(len(parcels))
            if plain[index] != scaled[index]
        }
        assert flipped
        for parcel in flipped:
            assert parcel.col in (2, 3, 4) and parcel.row in (0, 1, 2)

    def test_both_switches_agree_under_equal_weights(self) -> None:
        assert self._approvals_via_rewards(1, scale=True) == (
            self._approvals_via_consensus(1, weighted=True)
        )
        assert self._approvals_via_rewards(1, scale=True) == (
            self._approvals_via_rewards(1, scale=False)
        )
