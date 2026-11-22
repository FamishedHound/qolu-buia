"""A 50-iteration joint-loop smoke run of every QOLU reading.

Deliberately tiny - a synthetic 30 x 30 grid, a small decision buffer and a
batch of 16 - because the point is that the Rainbow wiring survives a real
BUIA-proposes / QOLU-appraises loop, not that anything is learned in 50
iterations.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chapter4_qolu.agent.buia_agent import BUIAAgent, BUIAConfig
from chapter4_qolu.agent.prioritised_replay import PrioritisedReplayBuffer
from chapter4_qolu.agent.qolu_agent import QOLUAgent, QOLUConfig
from chapter4_qolu.agent.qolu_variants import qolu_variant_settings
from chapter4_qolu.agent.replay import NStepReplayBuffer
from chapter4_qolu.config import SCENARIOS, EnvConfig
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index
from chapter4_qolu.runner import JointTrainingReport, train_joint

SMOKE_ITERATIONS = 50
GRID_SIZE = 30


def synthetic_grid(size: int = GRID_SIZE) -> ParcelGrid:
    """Grassland with a built-up seed corner and a bog block, as in the unit tests."""
    composition = np.zeros((N_AGGREGATE_CLASSES, size, size), dtype=np.float32)
    composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 1.0
    composition[:, :2, :2] = 0.0
    composition[band_index(AggregateClass.BUILT_UP), :2, :2] = 1.0
    composition[:, -6:, :4] = 0.0
    composition[band_index(AggregateClass.MOUNTAIN_HEATH_BOG), -6:, :4] = 1.0
    composition[:, :4, -6:] = 0.0
    composition[band_index(AggregateClass.BROADLEAF_WOODLAND), :4, -6:] = 1.0
    return ParcelGrid(composition)


def smoke_env() -> UrbanLandUseEnv:
    config = EnvConfig(
        grid_size=GRID_SIZE,
        samples_per_iteration=8,
        max_conversions_per_iteration=4,
        iterations=25,
        observation_window=5,
        buia_observation_window=5,
    )
    return UrbanLandUseEnv(synthetic_grid(), config=config, weights=SCENARIOS[1])


def smoke_run(variant: str, seed: int = 0) -> tuple[JointTrainingReport, QOLUAgent]:
    torch.manual_seed(seed)
    env = smoke_env()
    settings: dict[str, object] = {
        "observation_channels": env.observation_channels,
        "observation_window": env.config.observation_window,
        "batch_size": 16,
        "learning_starts": 16,
        "replay_capacity": 1_000,
        "target_sync_interval": 10,
        "n_step": 3,
        "epsilon_decay_steps": 100,
        "per_beta_steps": SMOKE_ITERATIONS,
    }
    settings.update(qolu_variant_settings(variant))
    stakeholder = QOLUAgent(QOLUConfig(**settings), seed=seed)  # type: ignore[arg-type]
    investor = BUIAAgent(
        BUIAConfig(
            observation_channels=env.buia_observation_channels,
            observation_window=env.config.buia_observation_window,
            max_proposals=env.config.max_conversions_per_iteration,
        ),
        seed=seed,
    )
    report = train_joint(
        env, [stakeholder], investor, iterations=SMOKE_ITERATIONS, seed=seed
    )
    return report, stakeholder


@pytest.fixture(scope="module")
def published() -> tuple[JointTrainingReport, QOLUAgent]:
    return smoke_run("published")


@pytest.fixture(scope="module")
def prioritised() -> tuple[JointTrainingReport, QOLUAgent]:
    return smoke_run("per")


@pytest.fixture(scope="module")
def c51() -> tuple[JointTrainingReport, QOLUAgent]:
    return smoke_run("c51")


Run = tuple[JointTrainingReport, QOLUAgent]


class TestPublishedSmoke:
    """``qolu_buia``: the chapter's reading - NoisyNet layers, uniform buffer."""

    def test_the_loop_completes(self, published: Run) -> None:
        report, _ = published
        assert report.iterations == SMOKE_ITERATIONS
        assert len(report.team_rewards) == SMOKE_ITERATIONS
        assert all(np.isfinite(value) for value in report.team_rewards)

    def test_gradient_steps_happened_and_the_losses_are_finite(
        self, published: Run
    ) -> None:
        report, agent = published
        assert report.qolu_losses
        assert all(np.isfinite(loss) for loss in report.qolu_losses)
        assert agent.gradient_steps == len(report.qolu_losses)

    def test_it_keeps_the_uniform_buffer_algorithm_1_describes(
        self, published: Run
    ) -> None:
        _, agent = published
        assert type(agent.replay) is NStepReplayBuffer

    def test_the_noisy_net_was_actually_exercised(self, published: Run) -> None:
        _, agent = published
        assert agent.config.noisy_nets is True
        sigma = agent.online_network.value_head.weight_sigma.detach()
        assert float(sigma.std()) > 0.0, "sigma never moved off its constant init"

    def test_the_landscape_urbanises(self, published: Run) -> None:
        report, _ = published
        assert sum(report.n_conversions) > 0
        assert report.final_metrics["urban"] >= 0.0

    def test_it_is_reproducible_from_its_seed(self) -> None:
        first, _ = smoke_run("published", seed=3)
        second, _ = smoke_run("published", seed=3)
        np.testing.assert_allclose(first.team_rewards, second.team_rewards, rtol=1e-6)
        np.testing.assert_allclose(first.qolu_losses, second.qolu_losses, rtol=1e-6)


class TestPrioritisedSmoke:
    """``qolu_buia_per``: the opt-in Rainbow extra."""

    def test_the_loop_completes(self, prioritised: Run) -> None:
        report, _ = prioritised
        assert report.iterations == SMOKE_ITERATIONS
        assert all(np.isfinite(value) for value in report.team_rewards)

    def test_gradient_steps_happened_and_the_losses_are_finite(
        self, prioritised: Run
    ) -> None:
        report, _ = prioritised
        assert report.qolu_losses
        assert all(np.isfinite(loss) for loss in report.qolu_losses)

    def test_the_prioritisation_was_actually_exercised(self, prioritised: Run) -> None:
        _, agent = prioritised
        assert isinstance(agent.replay, PrioritisedReplayBuffer)
        priorities = np.asarray(agent.replay.priorities())
        assert np.unique(priorities).size > 1, "priorities never moved off their max"
        assert agent.beta() > agent.config.per_beta_start


class TestC51Smoke:
    """``qolu_c51``: the plain C51 reading, neither component."""

    def test_the_loop_completes(self, c51: Run) -> None:
        report, _ = c51
        assert report.iterations == SMOKE_ITERATIONS
        assert all(np.isfinite(value) for value in report.team_rewards)

    def test_it_runs_the_uniform_buffer_and_plain_linear_layers(self, c51: Run) -> None:
        _, agent = c51
        assert type(agent.replay) is NStepReplayBuffer
        assert agent.config.noisy_nets is False

    def test_gradient_steps_happened_and_the_losses_are_finite(self, c51: Run) -> None:
        report, _ = c51
        assert report.qolu_losses
        assert all(np.isfinite(loss) for loss in report.qolu_losses)


class TestTheReadingsDiffer:
    def test_no_two_variants_are_the_same_run(
        self, published: Run, prioritised: Run, c51: Run
    ) -> None:
        losses = [run[0].qolu_losses for run in (published, prioritised, c51)]
        assert losses[0] != losses[1]
        assert losses[0] != losses[2]

    def test_every_variant_reports_the_same_published_metrics(
        self, published: Run, prioritised: Run, c51: Run
    ) -> None:
        metrics = [set(run[0].final_metrics) for run in (published, prioritised, c51)]
        assert metrics[0] == metrics[1] == metrics[2]
        assert metrics[0] >= {"land_use", "woodland", "urban"}
