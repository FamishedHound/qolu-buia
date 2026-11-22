"""Tests for the evaluation-rollout collector behind the distribution diagnostics.

Also pins the factual position the diagnostics state: the policy reads the
categorical distribution ``Z(s, a)`` **only** through its mean ``E[Z]``, so two
distributions with the same mean and different shape produce the same action
and the same quality score.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from chapter4_qolu.agent.buia_agent import BUIAAgent, BUIAConfig
from chapter4_qolu.agent.distributional import AtomSupport
from chapter4_qolu.agent.qolu_agent import QOLUAgent, QOLUConfig
from chapter4_qolu.config import SCENARIOS, EnvConfig
from chapter4_qolu.diagnostics.collect import collect_appraisals
from chapter4_qolu.diagnostics.records import DiagnosticsLog
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index

ITERATIONS = 4
SAMPLES = 6


def make_grid(size: int = 10) -> ParcelGrid:
    composition = np.zeros((N_AGGREGATE_CLASSES, size, size), dtype=np.float32)
    composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 1.0
    composition[:, 0, 0] = 0.0
    composition[band_index(AggregateClass.BUILT_UP), 0, 0] = 1.0
    composition[:, size - 3 :, :2] = 0.0
    composition[band_index(AggregateClass.BROADLEAF_WOODLAND), size - 3 :, :2] = 1.0
    return ParcelGrid(composition)


def make_env(**overrides: object) -> UrbanLandUseEnv:
    settings: dict[str, object] = {
        "grid_size": 10,
        "samples_per_iteration": SAMPLES,
        "max_conversions_per_iteration": 3,
        "iterations": 1000,
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
            n_step=2,
            n_atoms=11,
            v_min=-5.0,
            v_max=5.0,
        ),
        seed=seed,
    )


def make_investor(env: UrbanLandUseEnv, seed: int = 0) -> BUIAAgent:
    return BUIAAgent(
        BUIAConfig(
            observation_channels=env.buia_observation_channels,
            observation_window=env.config.buia_observation_window,
            max_proposals=3,
        ),
        seed=seed,
    )


def collect(n_stakeholders: int = 2) -> DiagnosticsLog:
    env = make_env()
    stakeholders = [
        make_stakeholder(env, seed=index) for index in range(n_stakeholders)
    ]
    return collect_appraisals(
        env,
        stakeholders,
        make_investor(env),
        iterations=ITERATIONS,
        seed=0,
    )


class TestCollectedShape:
    def test_one_log_per_stakeholder(self) -> None:
        log = collect(n_stakeholders=3)
        assert len(log.logs) == 3
        assert [entry.stakeholder for entry in log.logs] == [0, 1, 2]

    def test_one_row_per_appraised_candidate(self) -> None:
        log = collect()
        for entry in log.logs:
            assert entry.size == ITERATIONS * SAMPLES
            assert entry.probabilities.shape == (ITERATIONS * SAMPLES, 11)

    def test_every_row_is_a_probability_distribution(self) -> None:
        log = collect()
        for entry in log.logs:
            assert np.all(entry.probabilities >= 0.0)
            np.testing.assert_allclose(
                entry.probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6
            )

    def test_the_support_is_the_agents_own_atom_support(self) -> None:
        env = make_env()
        agent = make_stakeholder(env)
        log = collect_appraisals(env, [agent], None, iterations=2, seed=0)
        np.testing.assert_allclose(log.logs[0].atoms, agent.support.atoms().numpy())

    def test_the_predicted_mean_is_the_atom_weighted_average(self) -> None:
        log = collect()
        entry = log.logs[0]
        np.testing.assert_allclose(
            entry.predicted_mean, entry.probabilities @ entry.atoms, rtol=1e-6
        )

    def test_the_logged_action_is_one_of_the_two_appraisals(self) -> None:
        log = collect()
        for entry in log.logs:
            assert set(np.unique(entry.actions)).issubset({0, 1})

    def test_the_reference_network_is_logged_alongside(self) -> None:
        log = collect()
        for entry in log.logs:
            assert entry.reference_probabilities.shape == entry.probabilities.shape
            np.testing.assert_allclose(
                entry.reference_probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6
            )

    def test_the_metadata_records_the_reward_reading_and_the_horizon(self) -> None:
        log = collect()
        assert log.iterations == ITERATIONS
        assert log.qolu_reward_mode == "per_parcel"
        assert log.logs[0].gamma == pytest.approx(0.99)
        assert log.logs[0].n_step == 2


class TestRealisedReturns:
    def test_all_three_targets_are_offered_with_one_value_per_row(self) -> None:
        entry = collect().logs[0]
        targets = entry.realised_returns()
        assert set(targets) == {"n_step_reward_sum", "n_step_target", "tail_return"}
        for values in targets.values():
            assert values.shape == (ITERATIONS * SAMPLES,)

    def test_the_tail_return_of_the_last_row_is_its_own_reward(self) -> None:
        entry = collect().logs[0]
        assert entry.realised_returns()["tail_return"][-1] == pytest.approx(
            entry.rewards[-1]
        )

    def test_the_bootstrapped_target_differs_from_the_bootstrap_free_sum(self) -> None:
        entry = collect().logs[0]
        targets = entry.realised_returns()
        assert np.isfinite(entry.bootstrap_values).all()
        assert not np.array_equal(
            targets["n_step_target"], targets["n_step_reward_sum"]
        )


class TestTheCollectorDoesNotTrain:
    def test_no_gradient_step_is_taken(self) -> None:
        env = make_env()
        agent = make_stakeholder(env)
        collect_appraisals(env, [agent], None, iterations=ITERATIONS, seed=0)
        assert agent.gradient_steps == 0
        assert len(agent.replay) == 0

    def test_two_collections_with_the_same_seed_agree_exactly(self) -> None:
        first = collect()
        second = collect()
        for left, right in zip(first.logs, second.logs):
            np.testing.assert_array_equal(left.probabilities, right.probabilities)
            np.testing.assert_array_equal(left.rewards, right.rewards)


class _FixedDistribution(nn.Module):
    """A stand-in network that always emits one fixed distribution per action."""

    def __init__(self, probabilities: np.ndarray, support: AtomSupport) -> None:
        super().__init__()
        self.register_buffer(
            "fixed", torch.as_tensor(probabilities, dtype=torch.float32)
        )
        self.support = support

    def zero_noise(self) -> None:  # pragma: no cover - interface stub
        return None

    def reset_noise(self) -> None:  # pragma: no cover - interface stub
        return None

    def distribution(self, observations: torch.Tensor) -> torch.Tensor:
        return self.fixed.unsqueeze(0).expand(observations.shape[0], -1, -1)

    def q_values(self, observations: torch.Tensor) -> torch.Tensor:
        atoms = self.support.atoms(dtype=torch.float32)
        return (self.distribution(observations) * atoms).sum(dim=-1)


class TestThePolicyUsesOnlyTheMean:
    """``Z`` reaches the decision only through ``E[Z]``."""

    @staticmethod
    def _agent_with(probabilities: np.ndarray) -> QOLUAgent:
        agent = QOLUAgent(
            QOLUConfig(
                observation_channels=2,
                observation_window=3,
                n_atoms=3,
                v_min=-1.0,
                v_max=1.0,
            ),
            seed=0,
        )
        agent.online_network = _FixedDistribution(  # type: ignore[assignment]
            probabilities, agent.support
        )
        return agent

    #: Two shapes, one mean: ``E[Z] = (-0.5, +0.5)`` for (reject, approve) in both.
    NARROW = np.array([[0.75, 0.0, 0.25], [0.25, 0.0, 0.75]])
    WIDE = np.array([[0.7, 0.1, 0.2], [0.2, 0.1, 0.7]])

    def test_the_two_fixtures_really_do_share_their_means(self) -> None:
        atoms = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(self.NARROW @ atoms, self.WIDE @ atoms)
        assert not np.allclose(self.NARROW, self.WIDE)

    def test_the_greedy_action_is_unchanged_by_the_shape(self) -> None:
        observations = np.zeros((4, 2, 3, 3), dtype=np.float32)
        narrow = self._agent_with(self.NARROW).act(observations, epsilon=0.0)
        wide = self._agent_with(self.WIDE).act(observations, epsilon=0.0)
        np.testing.assert_array_equal(narrow, wide)

    def test_the_quality_score_is_unchanged_by_the_shape(self) -> None:
        observations = np.zeros((4, 2, 3, 3), dtype=np.float32)
        np.testing.assert_allclose(
            self._agent_with(self.NARROW).quality_scores(observations),
            self._agent_with(self.WIDE).quality_scores(observations),
            rtol=1e-6,
        )

    def test_the_agent_says_so_in_writing(self) -> None:
        from chapter4_qolu.agent import qolu_agent

        assert "only through its mean" in (qolu_agent.__doc__ or "")
