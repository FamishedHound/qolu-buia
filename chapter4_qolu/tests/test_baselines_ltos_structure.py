"""Structural tests for the LToS baseline: lattice, features and rewards.

Split from ``test_baselines_ltos.py`` (which covers the networks and the method)
to keep both files small.  The headline assertion lives here: the district
reward decomposition must sum to the published team reward.
"""

from __future__ import annotations

import numpy as np
import pytest

from chapter4_qolu.baselines.districts import (
    N_SLOTS,
    NODE_FEATURE_DIM,
    PARCEL_FEATURE_DIM,
    REVERSE_SLOT,
    DistrictLattice,
    incoming_weights,
    node_features,
    parcel_features,
)
from chapter4_qolu.baselines.ltos_rewards import (
    district_rewards,
    per_conversion_scores,
    shaped_rewards,
)
from chapter4_qolu.env.parcel_grid import ParcelIndex
from chapter4_qolu.env.rewards import compose_team_reward, objective_rewards
from chapter4_qolu.env.urban_env import APPROVE, REJECT
from chapter4_qolu.objectives import TopDownObjective
from chapter4_qolu.tests.baseline_fixtures import make_env


class TestDistrictLattice:
    def test_a_fifty_grid_gives_ten_by_ten_districts_of_five_by_five(self) -> None:
        lattice = DistrictLattice.for_grid(50, 50, districts_per_side=10)
        assert lattice.district_size == 5
        assert (lattice.n_rows, lattice.n_cols) == (10, 10)
        assert lattice.n_districts == 100

    def test_the_district_count_is_configurable(self) -> None:
        lattice = DistrictLattice.for_grid(50, 50, districts_per_side=5)
        assert lattice.district_size == 10
        assert lattice.n_districts == 25

    def test_a_grid_that_does_not_divide_evenly_keeps_a_smaller_last_district(
        self,
    ) -> None:
        lattice = DistrictLattice.for_grid(12, 12, districts_per_side=5)
        assert lattice.district_size == 3
        assert lattice.n_districts == 16
        assert lattice.district_of(ParcelIndex(11, 11)) == 15

    def test_parcels_map_to_the_district_that_contains_them(self) -> None:
        lattice = DistrictLattice.for_grid(10, 10, districts_per_side=5)
        assert lattice.district_of(ParcelIndex(0, 0)) == 0
        assert lattice.district_of(ParcelIndex(1, 1)) == 0
        assert lattice.district_of(ParcelIndex(0, 2)) == 1
        assert lattice.district_of(ParcelIndex(2, 0)) == 5

    def test_every_district_is_its_own_neighbour(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        for district in range(lattice.n_districts):
            assert lattice.neighbour_index[district, 0] == district
            assert lattice.neighbour_mask[district, 0] == 1.0

    def test_adjacency_is_rook_not_queen(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        centre = lattice.index_of(1, 1)
        neighbours = {
            int(index)
            for index, mask in zip(
                lattice.neighbour_index[centre], lattice.neighbour_mask[centre]
            )
            if mask > 0.0
        }
        assert neighbours == {
            centre,
            lattice.index_of(0, 1),
            lattice.index_of(2, 1),
            lattice.index_of(1, 0),
            lattice.index_of(1, 2),
        }

    def test_corner_districts_have_two_neighbours_plus_themselves(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        assert lattice.neighbour_mask[lattice.index_of(0, 0)].sum() == 3.0

    def test_reverse_slots_point_back_at_the_sender(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        for district in range(lattice.n_districts):
            for slot in range(N_SLOTS):
                if lattice.neighbour_mask[district, slot] == 0.0:
                    continue
                neighbour = int(lattice.neighbour_index[district, slot])
                back = REVERSE_SLOT[slot]
                assert int(lattice.neighbour_index[neighbour, back]) == district

    def test_rejects_a_non_positive_district_count(self) -> None:
        with pytest.raises(ValueError):
            DistrictLattice.for_grid(10, 10, districts_per_side=0)


class TestFeatures:
    def test_node_features_have_the_documented_width(self) -> None:
        env = make_env()
        env.reset(seed=0)
        lattice = DistrictLattice.for_grid(12, 12, districts_per_side=4)
        features = node_features(lattice, env.grid, env.attractiveness)
        assert features.shape == (lattice.n_districts, NODE_FEATURE_DIM)
        assert np.isfinite(features).all()

    def test_parcel_features_have_the_documented_width(self) -> None:
        env = make_env()
        batch = env.reset(seed=0)
        features = parcel_features(env.grid, env.attractiveness, batch.parcels)
        assert features.shape == (len(batch.parcels), PARCEL_FEATURE_DIM)
        assert np.isfinite(features).all()


class TestRewardDecomposition:
    """`sum_i r_i` must equal the published team reward - the note's condition."""

    def test_district_rewards_sum_to_the_published_team_reward(self) -> None:
        env = make_env(max_conversions_per_iteration=4)
        batch = env.reset(seed=3)
        grid_before = env.grid
        result = env.step(np.full(len(batch.parcels), APPROVE, dtype=np.int64))
        lattice = DistrictLattice.for_grid(12, 12, districts_per_side=4)
        rewards = district_rewards(
            lattice,
            grid_before,
            result.conversions,
            weights=env.weights,
            societal=result.societal_reward,
            config=env.config.reward,
        )
        assert rewards.shape == (lattice.n_districts,)
        assert float(rewards.sum()) == pytest.approx(result.team_reward, abs=1e-9)

    def test_the_equality_survives_an_iteration_with_no_conversions(self) -> None:
        env = make_env()
        batch = env.reset(seed=4)
        grid_before = env.grid
        result = env.step(np.full(len(batch.parcels), REJECT, dtype=np.int64))
        lattice = DistrictLattice.for_grid(12, 12, districts_per_side=4)
        rewards = district_rewards(
            lattice,
            grid_before,
            result.conversions,
            weights=env.weights,
            societal=result.societal_reward,
            config=env.config.reward,
        )
        assert float(rewards.sum()) == pytest.approx(result.team_reward, abs=1e-9)

    def test_the_equality_holds_under_every_published_scenario(self) -> None:
        for scenario in (1, 2, 3, 4):
            env = make_env(scenario=scenario, max_conversions_per_iteration=4)
            batch = env.reset(seed=5)
            grid_before = env.grid
            result = env.step(np.full(len(batch.parcels), APPROVE, dtype=np.int64))
            lattice = DistrictLattice.for_grid(12, 12, districts_per_side=4)
            rewards = district_rewards(
                lattice,
                grid_before,
                result.conversions,
                weights=env.weights,
                societal=result.societal_reward,
                config=env.config.reward,
            )
            assert float(rewards.sum()) == pytest.approx(result.team_reward, abs=1e-9)

    def test_only_districts_containing_a_conversion_earn_an_objective_term(
        self,
    ) -> None:
        env = make_env(max_conversions_per_iteration=1)
        env.reset(seed=6)
        grid_before = env.grid
        env.set_candidates([ParcelIndex(5, 5)])
        result = env.step([APPROVE])
        lattice = DistrictLattice.for_grid(12, 12, districts_per_side=4)
        rewards = district_rewards(
            lattice,
            grid_before,
            result.conversions,
            weights=env.weights,
            societal=0.0,
            config=env.config.reward,
        )
        owner = lattice.district_of(ParcelIndex(5, 5))
        assert rewards[owner] > 0.0
        assert np.count_nonzero(rewards) == 1

    def test_per_conversion_scores_average_to_the_published_objective_rewards(
        self,
    ) -> None:
        env = make_env(max_conversions_per_iteration=4)
        batch = env.reset(seed=7)
        grid_before = env.grid
        result = env.step(np.full(len(batch.parcels), APPROVE, dtype=np.int64))
        published = objective_rewards(
            grid_before, env.grid, result.conversions, env.config.reward
        )
        scores = per_conversion_scores(
            grid_before, result.conversions, env.config.reward
        )
        for objective in TopDownObjective:
            assert float(np.mean(scores[objective])) == pytest.approx(
                published[objective]
            )

    def test_the_societal_term_is_split_equally_across_districts(self) -> None:
        lattice = DistrictLattice.for_grid(12, 12, districts_per_side=4)
        env = make_env()
        env.reset(seed=8)
        rewards = district_rewards(
            lattice,
            env.grid,
            conversions=(),
            weights=env.weights,
            societal=1.0,
            config=env.config.reward,
        )
        np.testing.assert_allclose(rewards, 1.0 / lattice.n_districts)

    def test_composing_the_team_reward_by_hand_matches_the_environment(self) -> None:
        env = make_env(max_conversions_per_iteration=3)
        batch = env.reset(seed=9)
        grid_before = env.grid
        result = env.step(np.full(len(batch.parcels), APPROVE, dtype=np.int64))
        expected = compose_team_reward(
            objective_rewards(
                grid_before, env.grid, result.conversions, env.config.reward
            ),
            env.weights,
            result.societal_reward,
        )
        assert expected == pytest.approx(result.team_reward)


class TestShapedRewards:
    def test_sharing_is_conservative_when_every_row_sums_to_one(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        rng = np.random.default_rng(0)
        weights = rng.random((lattice.n_districts, N_SLOTS)) * lattice.neighbour_mask
        weights = weights / weights.sum(axis=1, keepdims=True)
        local = rng.normal(size=lattice.n_districts)
        shaped = shaped_rewards(local, weights, lattice)
        assert float(shaped.sum()) == pytest.approx(float(local.sum()))

    def test_a_fully_selfish_lattice_returns_the_local_rewards_unchanged(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        weights = np.zeros((lattice.n_districts, N_SLOTS))
        weights[:, 0] = 1.0
        local = np.arange(lattice.n_districts, dtype=np.float64)
        np.testing.assert_allclose(shaped_rewards(local, weights, lattice), local)

    def test_a_district_that_gives_everything_away_keeps_nothing(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        weights = np.zeros((lattice.n_districts, N_SLOTS))
        weights[:, 0] = 1.0
        centre = lattice.index_of(1, 1)
        east = lattice.index_of(1, 2)
        weights[centre] = 0.0
        weights[centre, 4] = 1.0  # east
        local = np.zeros(lattice.n_districts)
        local[centre] = 1.0
        shaped = shaped_rewards(local, weights, lattice)
        assert shaped[centre] == pytest.approx(0.0)
        assert shaped[east] == pytest.approx(1.0)


class TestIncomingWeights:
    def test_a_fully_selfish_lattice_sends_nothing_to_its_neighbours(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        outgoing = np.zeros((lattice.n_districts, N_SLOTS))
        outgoing[:, 0] = 1.0
        incoming = incoming_weights(outgoing, lattice)
        np.testing.assert_allclose(incoming[:, 0], 1.0)
        np.testing.assert_allclose(incoming[:, 1:], 0.0)

    def test_the_share_a_neighbour_sends_arrives_in_the_matching_slot(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        outgoing = np.zeros((lattice.n_districts, N_SLOTS))
        centre = lattice.index_of(1, 1)
        outgoing[centre, 4] = 1.0  # the centre sends everything east
        incoming = incoming_weights(outgoing, lattice)
        east = lattice.index_of(1, 2)
        assert incoming[east, 3] == pytest.approx(1.0)  # arrives from the west
        assert incoming[centre].sum() == pytest.approx(0.0)

    def test_absent_neighbours_contribute_nothing(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        outgoing = np.ones((lattice.n_districts, N_SLOTS))
        incoming = incoming_weights(outgoing, lattice)
        assert float((incoming * (1.0 - lattice.neighbour_mask)).sum()) == 0.0
