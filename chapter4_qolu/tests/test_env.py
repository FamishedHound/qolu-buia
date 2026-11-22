"""Tests for the POSG-style urban land-use environment (Section 3.5)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from chapter4_qolu.config import SCENARIOS, EnvConfig
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.env.rewards import TopDownObjective
from chapter4_qolu.env.urban_env import APPROVE, REJECT, UrbanLandUseEnv
from chapter4_qolu.land_cover import (
    N_AGGREGATE_CLASSES,
    AggregateClass,
    LandCoverGroup,
    band_index,
)


def make_grid(size: int = 12) -> ParcelGrid:
    """Grassland with a town, a wood and a bog block, so the three top-down
    objectives score differently from one another."""
    composition = np.zeros((N_AGGREGATE_CLASSES, size, size), dtype=np.float32)
    composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 1.0
    composition[:, 0, 0] = 0.0
    composition[band_index(AggregateClass.BUILT_UP), 0, 0] = 1.0
    composition[:, size - 1, size - 1] = 0.0
    composition[band_index(AggregateClass.BROADLEAF_WOODLAND), size - 1, size - 1] = 1.0
    composition[:, size - 4 :, :3] = 0.0
    composition[band_index(AggregateClass.MOUNTAIN_HEATH_BOG), size - 4 :, :3] = 1.0
    return ParcelGrid(composition)


def make_env(**overrides: object) -> UrbanLandUseEnv:
    settings: dict[str, object] = {
        "grid_size": 12,
        "samples_per_iteration": 6,
        "max_conversions_per_iteration": 3,
        "iterations": 5,
        "observation_window": 5,
    }
    settings.update(overrides)
    config = EnvConfig(**settings)  # type: ignore[arg-type]
    return UrbanLandUseEnv(make_grid(), config=config, weights=SCENARIOS[1])


class TestPublishedDefaults:
    def test_env_config_defaults_match_the_paper(self) -> None:
        config = EnvConfig()
        assert config.grid_size == 50  # thesis text: 50 x 50 parcels
        assert config.samples_per_iteration == 50
        assert config.iterations == 1000
        assert config.max_conversions_per_iteration == 10
        assert config.conversion_fraction == pytest.approx(1.0)

    def test_action_constants(self) -> None:
        assert REJECT == 0
        assert APPROVE == 1


class TestReset:
    def test_returns_a_decision_batch_of_the_configured_size(self) -> None:
        env = make_env()
        batch = env.reset(seed=0)
        assert len(batch.parcels) == 6
        assert batch.observations.shape == (6, env.observation_channels, 5, 5)
        assert batch.observations.dtype == np.float32

    def test_candidates_are_always_convertible_parcels(self) -> None:
        env = make_env()
        batch = env.reset(seed=1)
        convertible = env.grid.convertible_mask()
        assert all(convertible[parcel.row, parcel.col] for parcel in batch.parcels)

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        first = make_env().reset(seed=7)
        second = make_env().reset(seed=7)
        assert first.parcels == second.parcels
        np.testing.assert_allclose(first.observations, second.observations)

    def test_reset_restores_the_initial_landscape(self) -> None:
        env = make_env()
        env.reset(seed=0)
        env.step(np.ones(6, dtype=np.int64))
        env.reset(seed=0)
        np.testing.assert_allclose(env.grid.composition, make_grid().composition)
        assert env.iteration == 0

    def test_observation_channels_include_attractiveness(self) -> None:
        env = make_env()
        assert env.observation_channels == N_AGGREGATE_CLASSES + 1


class TestSynchronisedUpdate:
    def test_rejected_proposals_leave_the_landscape_untouched(self) -> None:
        env = make_env()
        env.reset(seed=2)
        before = env.grid.composition.copy()
        result = env.step(np.zeros(6, dtype=np.int64))
        np.testing.assert_allclose(env.grid.composition, before)
        assert result.conversions == ()

    def test_all_approvals_are_applied_in_one_synchronised_update(self) -> None:
        env = make_env(max_conversions_per_iteration=6)
        batch = env.reset(seed=3)
        result = env.step(np.ones(6, dtype=np.int64))
        unique_candidates = {parcel for parcel in batch.parcels}
        assert len(result.conversions) == len(unique_candidates)
        for conversion in result.conversions:
            urban = env.grid.composition[band_index(AggregateClass.BUILT_UP)]
            assert urban[conversion.parcel.row, conversion.parcel.col] == pytest.approx(
                1.0
            )

    def test_conversions_are_capped_per_iteration(self) -> None:
        env = make_env(samples_per_iteration=10, max_conversions_per_iteration=2)
        env.reset(seed=4)
        result = env.step(np.ones(10, dtype=np.int64))
        assert len(result.conversions) <= 2

    def test_duplicate_candidates_convert_only_once(self) -> None:
        env = make_env()
        env.reset(seed=5)
        # set_candidates is also the hook an external proposer can use.
        batch = env.set_candidates([ParcelIndex(5, 5), ParcelIndex(5, 5)])
        assert batch.observations.shape[0] == 2
        result = env.step(np.ones(2, dtype=np.int64))
        assert len(result.conversions) == 1

    def test_step_requires_one_decision_per_candidate(self) -> None:
        env = make_env()
        env.reset(seed=6)
        with pytest.raises(ValueError):
            env.step(np.ones(3, dtype=np.int64))

    def test_step_before_reset_raises(self) -> None:
        env = make_env()
        with pytest.raises(RuntimeError):
            env.step(np.zeros(6, dtype=np.int64))


class TestStepResult:
    def test_reports_shared_team_reward_and_per_objective_rewards(self) -> None:
        env = make_env()
        env.reset(seed=8)
        result = env.step(np.ones(6, dtype=np.int64))
        assert isinstance(result.team_reward, float)
        assert set(result.objective_rewards) == set(TopDownObjective)
        assert isinstance(result.societal_reward, float)

    def test_next_observations_describe_the_same_parcels_after_the_update(self) -> None:
        env = make_env()
        batch = env.reset(seed=9)
        result = env.step(np.ones(6, dtype=np.int64))
        assert result.next_observations.shape == batch.observations.shape
        assert result.parcels == batch.parcels

    def test_supplies_the_next_decision_batch(self) -> None:
        env = make_env()
        env.reset(seed=10)
        result = env.step(np.zeros(6, dtype=np.int64))
        assert len(result.next_batch.parcels) == 6

    def test_episode_terminates_after_the_configured_iterations(self) -> None:
        env = make_env(iterations=3)
        env.reset(seed=11)
        dones = []
        for _ in range(3):
            result = env.step(np.zeros(6, dtype=np.int64))
            dones.append(result.done)
        assert dones == [False, False, True]
        assert env.iteration == 3

    def test_attractiveness_is_updated_after_conversions(self) -> None:
        env = make_env(max_conversions_per_iteration=6)
        env.reset(seed=12)
        before = env.attractiveness.scores.copy()
        result = env.step(np.ones(6, dtype=np.int64))
        assert result.conversions  # sanity: something was converted
        assert not np.allclose(before, env.attractiveness.scores)

    def test_info_reports_the_conversion_alignment_metric(self) -> None:
        env = make_env(max_conversions_per_iteration=6)
        env.reset(seed=13)
        result = env.step(np.ones(6, dtype=np.int64))
        assert "conversion_alignment" in result.info
        assert result.info["conversion_alignment"] >= 0.0

    def test_urban_growth_is_monotonic_across_iterations(self) -> None:
        env = make_env(max_conversions_per_iteration=6, iterations=4)
        env.reset(seed=14)
        urban_share = []
        for _ in range(4):
            env.step(np.ones(6, dtype=np.int64))
            urban_share.append(
                float(env.grid.group_fraction(LandCoverGroup.URBAN).sum())
            )
        assert all(b >= a for a, b in pairwise(urban_share))


class TestScenarioWeighting:
    def test_weights_change_the_shared_reward(self) -> None:
        grid = make_grid()
        config = EnvConfig(
            grid_size=12,
            samples_per_iteration=4,
            max_conversions_per_iteration=4,
            iterations=5,
            observation_window=5,
        )
        rewards = []
        for scenario in (SCENARIOS[2], SCENARIOS[3]):
            env = UrbanLandUseEnv(grid, config=config, weights=scenario)
            env.reset(seed=21)
            rewards.append(env.step(np.ones(4, dtype=np.int64)).team_reward)
        assert rewards[0] != pytest.approx(rewards[1])

    def test_accepts_any_weight_triple(self) -> None:
        env = UrbanLandUseEnv(
            make_grid(),
            config=EnvConfig(
                grid_size=12, samples_per_iteration=2, observation_window=5
            ),
            weights=SCENARIOS[4],
        )
        env.reset(seed=22)
        assert env.weights.urban == pytest.approx(0.50)
