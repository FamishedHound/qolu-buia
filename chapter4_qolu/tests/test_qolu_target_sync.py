"""The target-network sync has to be *countable*, not only configured.

``QOLUConfig.target_sync_interval`` is an implementation choice - Algorithm 1
line 124 says "every ``t_n`` steps reset ``theta^- = theta``" and Table 4.1
never gives ``t_n``.  ``learn()`` fires once per iteration and only after
``learning_starts`` transitions have accumulated, so a 1000-iteration run takes
about 980 gradient steps, and an interval longer than that fires the reset zero
times.  A run in that state bootstraps its double-Q target off the *randomly
initialised* target network from beginning to end.

The config says what was asked for; only a counter says what happened, so the
counter is what these tests pin.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from chapter4_qolu.agent.checkpoint import (
    checkpoint_payload,
    in_memory_state,
    restore_in_memory,
)
from chapter4_qolu.agent.qolu_agent import QOLUAgent
from chapter4_qolu.config import QOLUConfig

SETTINGS: dict[str, Any] = {
    "observation_channels": 3,
    "observation_window": 3,
    "conv_channels": (4,),
    "hidden_units": 8,
    "n_atoms": 5,
    "v_min": -2.0,
    "v_max": 2.0,
    "batch_size": 4,
    "replay_capacity": 64,
    "learning_starts": 4,
    "n_step": 1,
    "target_sync_interval": 4,
    "noisy_nets": False,
}


def make_agent(**overrides: Any) -> QOLUAgent:
    return QOLUAgent(QOLUConfig(**{**SETTINGS, **overrides}), seed=0)


def fill(agent: QOLUAgent, n: int, rng: np.random.Generator) -> None:
    shape = (
        agent.config.observation_channels,
        agent.config.observation_window,
        agent.config.observation_window,
    )
    for _ in range(n):
        agent.remember(
            rng.random(shape, dtype=np.float32),
            int(rng.integers(0, 2)),
            float(rng.normal()),
            rng.random(shape, dtype=np.float32),
            False,
        )


class TestTheAgentCountsItsSyncs:
    def test_a_new_agent_has_synced_nothing(self) -> None:
        assert make_agent().target_syncs == 0

    def test_sync_target_increments_the_counter(self) -> None:
        agent = make_agent()
        agent.sync_target()
        agent.sync_target()
        assert agent.target_syncs == 2

    def test_learning_syncs_once_per_interval(self, rng: np.random.Generator) -> None:
        agent = make_agent(target_sync_interval=4)
        fill(agent, 32, rng)
        for _ in range(12):
            agent.learn()
        assert agent.gradient_steps == 12
        assert agent.target_syncs == 3

    def test_an_interval_longer_than_the_run_never_fires(
        self, rng: np.random.Generator
    ) -> None:
        """The failure mode this module exists for, in miniature."""
        agent = make_agent(target_sync_interval=1_000)
        fill(agent, 32, rng)
        for _ in range(12):
            agent.learn()
        assert agent.gradient_steps == 12
        assert agent.target_syncs == 0

    def test_a_target_that_never_synced_is_still_the_initialisation(
        self, rng: np.random.Generator
    ) -> None:
        """Zero syncs is not a bookkeeping detail: it is the whole bootstrap.

        With no sync the double-Q evaluation network keeps the weights it was
        constructed with, so ``theta^-`` is a random network for the entire
        run while ``theta`` moves away from it.
        """
        agent = make_agent(target_sync_interval=1_000)
        before = {
            name: tensor.clone()
            for name, tensor in agent.target_network.state_dict().items()
        }
        fill(agent, 32, rng)
        for _ in range(12):
            agent.learn()
        after = agent.target_network.state_dict()
        online = agent.online_network.state_dict()
        assert agent.target_syncs == 0
        assert all(torch.equal(before[name], after[name]) for name in before)
        assert not all(torch.equal(online[name], after[name]) for name in after)


class TestTheCampaignBudgetAtTheDefault:
    """The arithmetic of the published campaign, stated as a test.

    ``runner.train_joint`` calls ``agent.learn()`` once per iteration per
    stakeholder, and ``learn()`` returns without a gradient step until the
    buffer holds ``max(learning_starts, batch_size)`` transitions.
    """

    def test_a_1000_iteration_run_takes_about_980_gradient_steps(self) -> None:
        config = QOLUConfig(observation_channels=3, observation_window=9)
        samples_per_iteration, iterations = 50, 1_000
        warmup = -(
            -max(config.learning_starts, config.batch_size) // samples_per_iteration
        )
        gradient_steps = iterations - warmup
        assert warmup == 20
        assert gradient_steps == 980
        assert config.target_sync_interval == 25
        assert gradient_steps // config.target_sync_interval == 39


class TestTheCounterSurvivesACheckpoint:
    def test_the_in_memory_state_carries_it(self) -> None:
        agent = make_agent()
        agent.sync_target()
        assert in_memory_state(agent)["target_syncs"] == 1

    def test_it_round_trips_through_save_and_load(self, tmp_path: Path) -> None:
        agent = make_agent()
        for _ in range(3):
            agent.sync_target()
        path = agent.save(tmp_path / "stakeholder0.pt")
        assert QOLUAgent.load(path).target_syncs == 3

    def test_the_payload_of_a_trained_agent_states_the_interval_too(
        self, tmp_path: Path
    ) -> None:
        """A reader of the file alone must be able to check the arithmetic."""
        agent = make_agent(target_sync_interval=7)
        payload = checkpoint_payload(agent)
        assert payload["config"]["target_sync_interval"] == 7
        assert payload["target_syncs"] == 0

    def test_a_checkpoint_written_before_the_counter_existed_still_loads(
        self,
    ) -> None:
        """No version bump: a payload without the counter still loads.

        Such a payload simply has no ``target_syncs`` key, and reads as zero.
        """
        agent = make_agent()
        state = in_memory_state(agent)
        state.pop("target_syncs")
        restore_in_memory(agent, state)
        assert agent.target_syncs == 0

    def test_loading_does_not_silently_keep_the_live_counter(self) -> None:
        agent = make_agent()
        agent.sync_target()
        fresh = make_agent()
        fresh.load_state_dict(in_memory_state(agent))
        assert fresh.target_syncs == 1
        assert replace(agent.config).target_sync_interval == 4


@pytest.mark.parametrize("interval", [1, 3, 10])
def test_the_counter_matches_the_gradient_step_arithmetic(
    interval: int, rng: np.random.Generator
) -> None:
    agent = make_agent(target_sync_interval=interval)
    fill(agent, 32, rng)
    for _ in range(20):
        agent.learn()
    assert agent.target_syncs == agent.gradient_steps // interval
