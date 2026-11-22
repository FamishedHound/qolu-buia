"""Tests for the LToS baseline (Yi et al. 2021), mapping (A) of the note.

The lattice, feature and reward-decomposition tests live in
``test_baselines_ltos_structure.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chapter4_qolu.baselines.base import run_baseline
from chapter4_qolu.baselines.config import LToSConfig
from chapter4_qolu.baselines.districts import (
    N_SLOTS,
    NODE_FEATURE_DIM,
    PARCEL_FEATURE_DIM,
    DistrictLattice,
)
from chapter4_qolu.baselines.ltos import LToSMethod, fixed_ltos
from chapter4_qolu.baselines.ltos_networks import (
    DGNQNetwork,
    MultiHeadGraphAttention,
    SharingActor,
)
from chapter4_qolu.config import QOLUConfig
from chapter4_qolu.env.urban_env import APPROVE, REJECT
from chapter4_qolu.tests.baseline_fixtures import make_env


def make_method(env, seed: int = 0, **overrides: object) -> LToSMethod:
    settings: dict[str, object] = {
        "districts_per_side": 4,
        "hidden_units": 16,
        "attention_heads": 4,
        "attention_layers": 1,
        "batch_size": 4,
        "learning_starts": 4,
        "high_update_interval": 2,
        "high_action_interval": 2,
    }
    settings.update(overrides)
    return LToSMethod.for_env(env, seed=seed, **settings)  # type: ignore[arg-type]


class TestSharingActor:
    def test_rows_sum_to_one_over_the_present_neighbours(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        actor = SharingActor(node_dim=NODE_FEATURE_DIM, hidden_units=16)
        mask = torch.as_tensor(lattice.neighbour_mask, dtype=torch.float32)
        with torch.no_grad():
            weights = actor(torch.randn(2, lattice.n_districts, NODE_FEATURE_DIM), mask)
        np.testing.assert_allclose(weights.sum(dim=-1).numpy(), 1.0, atol=1e-6)

    def test_absent_neighbours_never_receive_weight(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        actor = SharingActor(node_dim=NODE_FEATURE_DIM, hidden_units=16)
        mask = torch.as_tensor(lattice.neighbour_mask, dtype=torch.float32)
        with torch.no_grad():
            weights = actor(torch.randn(1, lattice.n_districts, NODE_FEATURE_DIM), mask)
        assert float((weights.numpy() * (1.0 - lattice.neighbour_mask)).sum()) == 0.0

    def test_the_selfishness_initialiser_starts_interior_districts_at_the_target(
        self,
    ) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        actor = SharingActor(
            node_dim=NODE_FEATURE_DIM, hidden_units=16, selfishness=0.8
        )
        mask = torch.as_tensor(lattice.neighbour_mask, dtype=torch.float32)
        with torch.no_grad():
            weights = actor(torch.zeros(1, lattice.n_districts, NODE_FEATURE_DIM), mask)
        centre = lattice.index_of(1, 1)
        assert float(weights[0, centre, 0]) == pytest.approx(0.8, abs=1e-5)

    def test_boundary_districts_start_more_selfish_than_the_target(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        actor = SharingActor(
            node_dim=NODE_FEATURE_DIM, hidden_units=16, selfishness=0.8
        )
        mask = torch.as_tensor(lattice.neighbour_mask, dtype=torch.float32)
        with torch.no_grad():
            weights = actor(torch.zeros(1, lattice.n_districts, NODE_FEATURE_DIM), mask)
        assert float(weights[0, lattice.index_of(0, 0), 0]) > 0.8

    def test_the_remainder_is_split_evenly_over_the_neighbours(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        actor = SharingActor(
            node_dim=NODE_FEATURE_DIM, hidden_units=16, selfishness=0.8
        )
        mask = torch.as_tensor(lattice.neighbour_mask, dtype=torch.float32)
        with torch.no_grad():
            weights = actor(torch.zeros(1, lattice.n_districts, NODE_FEATURE_DIM), mask)
        centre = lattice.index_of(1, 1)
        shares = weights[0, centre, 1:].numpy()
        np.testing.assert_allclose(shares, 0.05, atol=1e-5)


class TestDGNNetwork:
    def test_returns_two_q_values_per_candidate_parcel(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        network = DGNQNetwork(
            node_dim=NODE_FEATURE_DIM,
            parcel_dim=PARCEL_FEATURE_DIM,
            hidden_units=16,
            heads=4,
            attention_layers=2,
        )
        nodes = torch.randn(3, lattice.n_districts, NODE_FEATURE_DIM)
        w_in = torch.rand(3, lattice.n_districts, N_SLOTS)
        parcels = torch.randn(3, 7, PARCEL_FEATURE_DIM)
        districts = torch.randint(0, lattice.n_districts, (3, 7))
        values = network(
            nodes,
            w_in,
            parcels,
            districts,
            torch.as_tensor(lattice.neighbour_index),
            torch.as_tensor(lattice.neighbour_mask, dtype=torch.float32),
        )
        assert values.shape == (3, 7, 2)

    def test_the_q_function_depends_on_the_incoming_sharing_weights(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        torch.manual_seed(0)
        network = DGNQNetwork(
            node_dim=NODE_FEATURE_DIM,
            parcel_dim=PARCEL_FEATURE_DIM,
            hidden_units=16,
            heads=4,
            attention_layers=1,
        )
        nodes = torch.randn(1, lattice.n_districts, NODE_FEATURE_DIM)
        parcels = torch.randn(1, 4, PARCEL_FEATURE_DIM)
        districts = torch.zeros(1, 4, dtype=torch.long)
        index = torch.as_tensor(lattice.neighbour_index)
        mask = torch.as_tensor(lattice.neighbour_mask, dtype=torch.float32)
        with torch.no_grad():
            first = network(
                nodes,
                torch.zeros(1, lattice.n_districts, N_SLOTS),
                parcels,
                districts,
                index,
                mask,
            )
            second = network(
                nodes,
                torch.ones(1, lattice.n_districts, N_SLOTS),
                parcels,
                districts,
                index,
                mask,
            )
        assert not torch.allclose(first, second)

    def test_attention_only_mixes_over_the_rook_neighbourhood(self) -> None:
        lattice = DistrictLattice.for_grid(9, 9, districts_per_side=3)
        torch.manual_seed(1)
        attention = MultiHeadGraphAttention(dim=16, heads=4)
        index = torch.as_tensor(lattice.neighbour_index)
        mask = torch.as_tensor(lattice.neighbour_mask, dtype=torch.float32)
        features = torch.randn(1, lattice.n_districts, 16)
        corner = lattice.index_of(0, 0)

        with torch.no_grad():
            baseline = attention(features, index, mask)
            far = features.clone()
            far[0, lattice.index_of(2, 2)] += 5.0  # not adjacent to the corner
            unchanged = attention(far, index, mask)
            near = features.clone()
            near[0, lattice.index_of(0, 1)] += 5.0  # the corner's eastern neighbour
            changed = attention(near, index, mask)

        torch.testing.assert_close(baseline[0, corner], unchanged[0, corner])
        assert not torch.allclose(baseline[0, corner], changed[0, corner])

    def test_hidden_units_must_divide_by_the_head_count(self) -> None:
        with pytest.raises(ValueError):
            MultiHeadGraphAttention(dim=10, heads=4)


class TestLToSConfig:
    def test_gamma_comes_from_the_published_chapter_four_configuration(self) -> None:
        assert LToSConfig().gamma == QOLUConfig.gamma

    def test_the_published_selfishness_initialiser_is_zero_point_eight(self) -> None:
        assert LToSConfig().selfishness == 0.8

    def test_the_traffic_column_hyper_parameters_are_the_defaults(self) -> None:
        config = LToSConfig()
        assert config.buffer_capacity == 10_000
        assert config.batch_size == 20
        assert config.tau == 0.1
        assert config.epsilon_start == 0.4
        assert config.epsilon_decay == 0.9
        assert config.epsilon_end == 0.05
        assert config.ou_sigma == 0.25
        assert config.high_action_interval == 15

    def test_rejects_a_non_positive_district_count(self) -> None:
        with pytest.raises(ValueError):
            LToSConfig(districts_per_side=0)


class TestLToSMethod:
    def test_reports_its_published_name(self) -> None:
        assert make_method(make_env()).name == "ltos"

    def test_one_agent_per_district(self) -> None:
        env = make_env()
        method = make_method(env)
        method.reset(env, 0)
        assert method.lattice is not None
        assert method.lattice.n_districts == 16

    def test_decisions_cover_every_candidate(self) -> None:
        env = make_env()
        method = make_method(env)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        decision = method.decide(batch)
        assert decision.decisions.shape == (len(batch.parcels),)
        assert set(np.unique(decision.decisions)) <= {REJECT, APPROVE}

    def test_never_approves_more_than_the_shared_conversion_budget(self) -> None:
        env = make_env(max_conversions_per_iteration=2, samples_per_iteration=8)
        method = make_method(env, epsilon_start=1.0)
        batch = env.reset(seed=1)
        method.reset(env, 1)
        for _ in range(5):
            decision = method.decide(batch)
            approved = {
                parcel
                for parcel, action in zip(batch.parcels, decision.decisions)
                if action == APPROVE
            }
            assert len(approved) <= 2
            result = env.step(decision.decisions)
            method.observe(result)
            batch = result.next_batch

    def test_an_unbounded_budget_lets_every_chosen_parcel_through(self) -> None:
        env = make_env(max_conversions_per_iteration=None)
        method = make_method(env, epsilon_start=1.0)
        batch = env.reset(seed=2)
        method.reset(env, 2)
        decision = method.decide(batch)
        assert decision.decisions.sum() >= 1

    def test_sharing_weights_are_a_row_stochastic_matrix(self) -> None:
        env = make_env()
        method = make_method(env)
        batch = env.reset(seed=3)
        method.reset(env, 3)
        method.decide(batch)
        weights = method.sharing_weights
        assert weights is not None
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
        assert float((weights * (1.0 - method.lattice.neighbour_mask)).sum()) == 0.0

    def test_the_high_level_acts_only_every_m_steps(self) -> None:
        env = make_env()
        method = make_method(env, high_action_interval=3, ou_sigma=0.25)
        batch = env.reset(seed=4)
        method.reset(env, 4)
        snapshots = []
        for _ in range(4):
            method.decide(batch)
            snapshots.append(method.sharing_weights.copy())
            result = env.step(np.zeros(len(batch.parcels), dtype=np.int64))
            method.observe(result)
            batch = result.next_batch
        np.testing.assert_allclose(snapshots[0], snapshots[1])
        np.testing.assert_allclose(snapshots[1], snapshots[2])
        assert not np.allclose(snapshots[2], snapshots[3])

    def test_learns_and_reduces_its_low_level_loss(self) -> None:
        env = make_env()
        method = make_method(env, batch_size=4, learning_starts=4)
        report = run_baseline(env, method, iterations=30, seed=5)
        assert report.method == "ltos"
        assert len(report.losses) >= 20
        assert np.mean(report.losses[-5:]) < np.mean(report.losses[:5]) * 5.0

    def test_the_sharing_actor_moves_when_learning_is_enabled(self) -> None:
        env = make_env()
        method = make_method(
            env, high_update_interval=1, learning_starts=4, batch_size=4
        )
        before = [p.detach().clone() for p in method.actor.parameters()]
        run_baseline(env, method, iterations=20, seed=6)
        assert any(
            not torch.allclose(a, b) for a, b in zip(before, method.actor.parameters())
        )

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        def run() -> tuple[float, ...]:
            env = make_env()
            return run_baseline(env, make_method(env, seed=7), 12, seed=7).team_rewards

        np.testing.assert_allclose(run(), run())


class TestFixedLToS:
    def test_reports_a_distinct_name(self) -> None:
        env = make_env()
        assert fixed_ltos(env, seed=0, districts_per_side=4).name == "fixed_ltos"

    def test_the_configuration_records_that_sharing_is_frozen(self) -> None:
        env = make_env()
        assert (
            fixed_ltos(env, seed=0, districts_per_side=4).config.learn_sharing is False
        )

    def test_sharing_weights_stay_at_the_selfishness_initialiser(self) -> None:
        env = make_env()
        method = fixed_ltos(
            env,
            seed=0,
            districts_per_side=4,
            hidden_units=16,
            attention_heads=4,
            attention_layers=1,
            batch_size=4,
            learning_starts=4,
        )
        batch = env.reset(seed=8)
        method.reset(env, 8)
        method.decide(batch)
        first = method.sharing_weights.copy()
        before = [p.detach().clone() for p in method.actor.parameters()]
        run_baseline(env, method, iterations=15, seed=8)
        assert all(
            torch.allclose(a, b) for a, b in zip(before, method.actor.parameters())
        )
        method.reset(env, 8)
        method.decide(env.reset(seed=8))
        np.testing.assert_allclose(first, method.sharing_weights)

    def test_no_exploration_noise_is_added_to_the_frozen_weights(self) -> None:
        env = make_env()
        method = fixed_ltos(
            env, seed=0, districts_per_side=4, hidden_units=16, attention_heads=4
        )
        batch = env.reset(seed=9)
        method.reset(env, 9)
        method.decide(batch)
        interior = method.lattice.index_of(1, 1)
        assert float(method.sharing_weights[interior, 0]) == pytest.approx(
            0.8, abs=1e-5
        )

    def test_the_low_level_policy_still_learns(self) -> None:
        env = make_env()
        method = fixed_ltos(
            env,
            seed=0,
            districts_per_side=4,
            hidden_units=16,
            attention_heads=4,
            attention_layers=1,
            batch_size=4,
            learning_starts=4,
        )
        before = [p.detach().clone() for p in method.q_network.parameters()]
        run_baseline(env, method, iterations=20, seed=10)
        assert any(
            not torch.allclose(a, b)
            for a, b in zip(before, method.q_network.parameters())
        )
