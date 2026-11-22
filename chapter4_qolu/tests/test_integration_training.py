"""End-to-end integration test: UKCEH raster -> environment -> QOLU training.

This exercises the whole Chapter 4 pipeline on the Northern-Ireland placeholder
raster, a stand-in for the paper's study region.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch

from chapter4_qolu.agent.qolu_agent import QOLUAgent
from chapter4_qolu.config import SCENARIOS, EnvConfig, QOLUConfig
from chapter4_qolu.data.lcm_loader import load_lcm_raster, select_region_of_interest
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.runner import TrainingReport, train_qolu


@pytest.fixture()
def small_env(lcm_tif_path: Path) -> UrbanLandUseEnv:
    raster = load_lcm_raster(lcm_tif_path, warn_on_placeholder=False)
    grid = select_region_of_interest(raster, grid_size=16)
    config = EnvConfig(
        grid_size=16,
        samples_per_iteration=8,
        max_conversions_per_iteration=4,
        iterations=6,
        observation_window=5,
    )
    return UrbanLandUseEnv(grid, config=config, weights=SCENARIOS[1])


def make_agent(env: UrbanLandUseEnv) -> QOLUAgent:
    config = QOLUConfig(
        observation_channels=env.observation_channels,
        observation_window=env.config.observation_window,
        n_actions=2,
        batch_size=16,
        learning_starts=16,
        replay_capacity=500,
        target_sync_interval=5,
        n_step=2,
        epsilon_decay_steps=20,
    )
    return QOLUAgent(config, seed=1)


class TestEndToEndTraining:
    def test_environment_builds_from_the_real_raster(
        self, small_env: UrbanLandUseEnv
    ) -> None:
        batch = small_env.reset(seed=0)
        assert batch.observations.shape == (8, small_env.observation_channels, 5, 5)
        assert np.all(np.isfinite(batch.observations))
        assert small_env.grid.land_mask().any()

    def test_training_runs_without_error_and_learns(
        self, small_env: UrbanLandUseEnv
    ) -> None:
        agent = make_agent(small_env)
        before = [p.detach().clone() for p in agent.online_network.parameters()]

        report = train_qolu(small_env, agent, iterations=24, seed=3)

        assert isinstance(report, TrainingReport)
        assert report.iterations == 24
        assert len(report.team_rewards) == 24
        assert all(np.isfinite(reward) for reward in report.team_rewards)
        assert report.losses, "expected at least one gradient step"
        assert all(np.isfinite(loss) for loss in report.losses)
        after = list(agent.online_network.parameters())
        assert any(not torch.allclose(a, b) for a, b in zip(before, after))

    def test_training_records_land_use_diagnostics(
        self, small_env: UrbanLandUseEnv
    ) -> None:
        agent = make_agent(small_env)
        report = train_qolu(small_env, agent, iterations=12, seed=4)
        assert len(report.urban_fraction) == 12
        assert len(report.conversion_alignment) == 12
        assert all(0.0 <= value <= 1.0 for value in report.urban_fraction)
        # Land conversion is one-way, so urban cover grows monotonically *within*
        # an episode (a reset restores the initial landscape).
        episode_length = small_env.config.iterations
        within_first_episode = report.urban_fraction[:episode_length]
        assert all(b >= a for a, b in pairwise(within_first_episode))
        assert within_first_episode[-1] > within_first_episode[0]

    def test_training_is_reproducible(self, small_env: UrbanLandUseEnv) -> None:
        first = train_qolu(small_env, make_agent(small_env), iterations=10, seed=5)
        second = train_qolu(small_env, make_agent(small_env), iterations=10, seed=5)
        np.testing.assert_allclose(first.team_rewards, second.team_rewards, rtol=1e-6)

    def test_episode_boundaries_are_handled(self, small_env: UrbanLandUseEnv) -> None:
        # 14 iterations with an episode length of 6 forces two resets mid-run.
        agent = make_agent(small_env)
        report = train_qolu(small_env, agent, iterations=14, seed=6)
        assert report.episodes >= 2
