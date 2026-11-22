"""Tests for QOLU checkpoint persistence.

A distribution diagnostic can only be run on a *trained* agent, so the agent
has to survive the end of the run it was trained in.  The round-trip has to be
exact, not approximate: the whole point of the diagnostic is to score the
categorical distribution the trained network emits, so a checkpoint that
restored it to five decimal places would put a floor under every calibration
statistic that has nothing to do with the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from chapter4_qolu.agent.buia_agent import BUIAAgent, BUIAConfig
from chapter4_qolu.agent.checkpoint import (
    CHECKPOINT_VERSION,
    load_agents,
    read_checkpoint,
    save_agents,
)
from chapter4_qolu.agent.qolu_agent import QOLUAgent, QOLUConfig
from chapter4_qolu.config import SCENARIOS, EnvConfig
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index
from chapter4_qolu.runner import train_joint


def make_agent(seed: int = 0, **overrides: object) -> QOLUAgent:
    settings: dict[str, object] = {
        "observation_channels": 4,
        "observation_window": 5,
        "batch_size": 8,
        "learning_starts": 8,
        "replay_capacity": 200,
        "target_sync_interval": 4,
        "n_step": 2,
        "n_atoms": 11,
        "v_min": -5.0,
        "v_max": 5.0,
    }
    settings.update(overrides)
    return QOLUAgent(QOLUConfig(**settings), seed=seed)  # type: ignore[arg-type]


def observations(count: int = 6) -> np.ndarray:
    rng = np.random.default_rng(12345)
    return rng.random((count, 4, 5, 5)).astype(np.float32)


def train_a_little(agent: QOLUAgent) -> None:
    """A handful of gradient steps, so the checkpoint is not the initialiser."""
    obs = observations(40)
    for index in range(39):
        agent.remember(obs[index], index % 2, float(index % 3), obs[index + 1], False)
    for _ in range(3):
        agent.learn()


class TestRoundTrip:
    def test_save_writes_a_single_file(self, tmp_path: Path) -> None:
        agent = make_agent()
        path = agent.save(tmp_path / "agent.pt")
        assert path.is_file()

    def test_a_loaded_agent_emits_bit_identical_distributions(
        self, tmp_path: Path
    ) -> None:
        agent = make_agent()
        train_a_little(agent)
        obs = observations(5)
        before = agent.action_distributions(obs)

        restored = QOLUAgent.load(agent.save(tmp_path / "agent.pt"))
        np.testing.assert_array_equal(restored.action_distributions(obs), before)

    def test_a_loaded_agent_emits_bit_identical_quality_scores(
        self, tmp_path: Path
    ) -> None:
        agent = make_agent()
        train_a_little(agent)
        obs = observations(5)
        before = agent.quality_scores(obs)

        restored = QOLUAgent.load(agent.save(tmp_path / "agent.pt"))
        np.testing.assert_array_equal(restored.quality_scores(obs), before)

    def test_the_target_network_is_restored_too(self, tmp_path: Path) -> None:
        agent = make_agent()
        train_a_little(agent)
        restored = QOLUAgent.load(agent.save(tmp_path / "agent.pt"))
        for saved, loaded in zip(
            agent.target_network.state_dict().values(),
            restored.target_network.state_dict().values(),
        ):
            torch.testing.assert_close(saved, loaded, rtol=0.0, atol=0.0)

    def test_the_config_survives_the_round_trip(self, tmp_path: Path) -> None:
        agent = make_agent(n_atoms=17, v_min=-3.0, v_max=4.0)
        restored = QOLUAgent.load(agent.save(tmp_path / "agent.pt"))
        assert restored.config == agent.config

    def test_the_atom_support_survives_the_round_trip(self, tmp_path: Path) -> None:
        agent = make_agent(n_atoms=17, v_min=-3.0, v_max=4.0)
        restored = QOLUAgent.load(agent.save(tmp_path / "agent.pt"))
        assert restored.support == agent.support
        np.testing.assert_array_equal(
            restored.support.atoms().numpy(), agent.support.atoms().numpy()
        )

    def test_the_step_counters_survive_the_round_trip(self, tmp_path: Path) -> None:
        agent = make_agent()
        train_a_little(agent)
        agent.act(observations(7), epsilon=0.0)
        restored = QOLUAgent.load(agent.save(tmp_path / "agent.pt"))
        assert restored.steps == agent.steps
        assert restored.gradient_steps == agent.gradient_steps

    def test_the_numpy_rng_stream_survives_the_round_trip(self, tmp_path: Path) -> None:
        agent = make_agent()
        agent.rng.random(17)  # move the stream off its seeded start
        restored = QOLUAgent.load(agent.save(tmp_path / "agent.pt"))
        np.testing.assert_array_equal(restored.rng.random(5), agent.rng.random(5))

    def test_the_torch_rng_state_survives_the_round_trip(self, tmp_path: Path) -> None:
        agent = make_agent()
        torch.manual_seed(99)
        restored = QOLUAgent.load(agent.save(tmp_path / "agent.pt"))
        torch.manual_seed(1234)  # clobber the global stream before restoring it
        restored.restore_torch_rng()
        drawn = torch.randn(4)
        torch.manual_seed(99)
        torch.testing.assert_close(drawn, torch.randn(4), rtol=0.0, atol=0.0)


class TestCheckpointPayload:
    def test_the_payload_is_versioned(self, tmp_path: Path) -> None:
        agent = make_agent()
        payload = read_checkpoint(agent.save(tmp_path / "agent.pt"))
        assert payload["version"] == CHECKPOINT_VERSION

    def test_the_payload_carries_the_support_explicitly(self, tmp_path: Path) -> None:
        agent = make_agent(n_atoms=13, v_min=-2.0, v_max=6.0)
        payload = read_checkpoint(agent.save(tmp_path / "agent.pt"))
        assert payload["support"] == {"n_atoms": 13, "v_min": -2.0, "v_max": 6.0}

    def test_an_unknown_version_is_refused(self, tmp_path: Path) -> None:
        agent = make_agent()
        path = agent.save(tmp_path / "agent.pt")
        payload = read_checkpoint(path)
        payload["version"] = CHECKPOINT_VERSION + 99
        torch.save(payload, path)
        with pytest.raises(ValueError, match="checkpoint format"):
            QOLUAgent.load(path)


class TestAgentDirectories:
    def test_save_agents_writes_one_file_per_stakeholder(self, tmp_path: Path) -> None:
        agents = [make_agent(seed=index) for index in range(3)]
        paths = save_agents(agents, tmp_path / "agents")
        assert [path.name for path in paths] == [
            "stakeholder0.pt",
            "stakeholder1.pt",
            "stakeholder2.pt",
        ]

    def test_load_agents_round_trips_the_whole_set_in_order(
        self, tmp_path: Path
    ) -> None:
        agents = [make_agent(seed=index) for index in range(3)]
        for agent in agents:
            train_a_little(agent)
        save_agents(agents, tmp_path / "agents")
        obs = observations(4)
        restored = load_agents(tmp_path / "agents")
        assert len(restored) == 3
        for loaded, original in zip(restored, agents):
            np.testing.assert_array_equal(
                loaded.action_distributions(obs), original.action_distributions(obs)
            )

    def test_load_agents_refuses_an_empty_directory(self, tmp_path: Path) -> None:
        (tmp_path / "agents").mkdir()
        with pytest.raises(FileNotFoundError):
            load_agents(tmp_path / "agents")


def make_grid(size: int = 10) -> ParcelGrid:
    composition = np.zeros((N_AGGREGATE_CLASSES, size, size), dtype=np.float32)
    composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 1.0
    composition[:, 0, 0] = 0.0
    composition[band_index(AggregateClass.BUILT_UP), 0, 0] = 1.0
    return ParcelGrid(composition)


def make_env(**overrides: object) -> UrbanLandUseEnv:
    settings: dict[str, object] = {
        "grid_size": 10,
        "samples_per_iteration": 6,
        "max_conversions_per_iteration": 3,
        "iterations": 4,
        "observation_window": 5,
        "buia_observation_window": 5,
    }
    settings.update(overrides)
    return UrbanLandUseEnv(
        make_grid(),
        config=EnvConfig(**settings),  # type: ignore[arg-type]
        weights=SCENARIOS[1],
    )


def make_stakeholder(env: UrbanLandUseEnv, seed: int = 0) -> QOLUAgent:
    return QOLUAgent(
        QOLUConfig(
            observation_channels=env.observation_channels,
            observation_window=env.config.observation_window,
            batch_size=8,
            learning_starts=8,
            replay_capacity=200,
            target_sync_interval=5,
            n_step=2,
            n_atoms=11,
            v_min=-5.0,
            v_max=5.0,
        ),
        seed=seed,
    )


class TestTrainingEntryPointPersistsAgents:
    def test_train_joint_writes_a_checkpoint_per_stakeholder(
        self, tmp_path: Path
    ) -> None:
        env = make_env()
        stakeholders = [make_stakeholder(env, seed=index) for index in range(2)]
        investor = BUIAAgent(
            BUIAConfig(
                observation_channels=env.buia_observation_channels,
                observation_window=env.config.buia_observation_window,
                max_proposals=3,
            ),
            seed=0,
        )
        train_joint(
            env,
            stakeholders,
            investor,
            iterations=4,
            seed=0,
            checkpoint_dir=tmp_path / "agents",
        )
        written = sorted(path.name for path in (tmp_path / "agents").glob("*.pt"))
        assert written == ["stakeholder0.pt", "stakeholder1.pt"]

    def test_the_persisted_agents_match_the_agents_the_run_ended_with(
        self, tmp_path: Path
    ) -> None:
        env = make_env()
        stakeholders = [make_stakeholder(env, seed=0)]
        train_joint(
            env,
            stakeholders,
            None,
            iterations=4,
            seed=0,
            checkpoint_dir=tmp_path / "agents",
        )
        obs = observations(3)[:, :1]
        obs = np.repeat(obs, env.observation_channels, axis=1)
        restored = load_agents(tmp_path / "agents")[0]
        np.testing.assert_array_equal(
            restored.action_distributions(obs),
            stakeholders[0].action_distributions(obs),
        )

    def test_no_checkpoint_directory_means_nothing_is_written(
        self, tmp_path: Path
    ) -> None:
        env = make_env()
        train_joint(env, [make_stakeholder(env)], None, iterations=3, seed=0)
        assert not list(tmp_path.glob("**/*.pt"))
