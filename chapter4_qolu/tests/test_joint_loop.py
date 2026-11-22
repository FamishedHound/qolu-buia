"""Tests for the joint BUIA-proposes / QOLU-appraises iteration (Section 3.5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chapter4_qolu.agent.buia_agent import BUIAAgent, BUIAConfig
from chapter4_qolu.agent.consensus import action_weighted_votes, consensus_decisions
from chapter4_qolu.agent.qolu_agent import QOLUAgent, QOLUConfig
from chapter4_qolu.config import SCENARIOS, EnvConfig, QOLURewardConfig
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.env.urban_env import APPROVE, REJECT, UrbanLandUseEnv
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index
from chapter4_qolu.objectives import TopDownObjective
from chapter4_qolu.runner import JointTrainingReport, train_joint


def make_grid(size: int = 12) -> ParcelGrid:
    composition = np.zeros((N_AGGREGATE_CLASSES, size, size), dtype=np.float32)
    composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 1.0
    composition[:, 0, 0] = 0.0
    composition[band_index(AggregateClass.BUILT_UP), 0, 0] = 1.0
    composition[:, size - 4 :, :3] = 0.0
    composition[band_index(AggregateClass.MOUNTAIN_HEATH_BOG), size - 4 :, :3] = 1.0
    return ParcelGrid(composition)


def make_env(**overrides: object) -> UrbanLandUseEnv:
    settings: dict[str, object] = {
        "grid_size": 12,
        "samples_per_iteration": 8,
        "max_conversions_per_iteration": 4,
        "iterations": 6,
        "observation_window": 5,
        "buia_observation_window": 5,
    }
    settings.update(overrides)
    return UrbanLandUseEnv(
        make_grid(), config=EnvConfig(**settings), weights=SCENARIOS[1]  # type: ignore[arg-type]
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
            epsilon_decay_steps=20,
        ),
        seed=seed,
    )


def make_investor(env: UrbanLandUseEnv, seed: int = 0) -> BUIAAgent:
    return BUIAAgent(
        BUIAConfig(
            observation_channels=env.buia_observation_channels,
            observation_window=env.config.buia_observation_window,
            max_proposals=env.config.max_conversions_per_iteration,
        ),
        seed=seed,
    )


class TestBUIAObservations:
    def test_two_km_radius_is_a_five_by_five_window(self) -> None:
        assert EnvConfig().buia_observation_window == 5

    def test_channels_add_a_land_use_history_plane(self) -> None:
        env = make_env()
        assert env.buia_observation_channels == N_AGGREGATE_CLASSES + 2
        assert env.buia_observation_channels == env.observation_channels + 1

    def test_observation_shape(self) -> None:
        env = make_env()
        batch = env.reset(seed=0)
        observations = env.buia_observations(batch.parcels)
        assert observations.shape == (
            len(batch.parcels),
            env.buia_observation_channels,
            5,
            5,
        )
        assert observations.dtype == np.float32

    def test_history_plane_records_past_conversions(self) -> None:
        env = make_env()
        env.reset(seed=0)
        parcel = ParcelIndex(5, 5)
        assert env.buia_observations([parcel])[0, -1, 2, 2] == pytest.approx(0.0)
        env.set_candidates([parcel])
        env.step([APPROVE])
        assert env.buia_observations([parcel])[0, -1, 2, 2] == pytest.approx(1.0)

    def test_history_is_cleared_on_reset(self) -> None:
        env = make_env()
        env.reset(seed=0)
        env.set_candidates([ParcelIndex(5, 5)])
        env.step([APPROVE])
        env.reset(seed=0)
        assert env.buia_observations([ParcelIndex(5, 5)])[0, -1, 2, 2] == (
            pytest.approx(0.0)
        )


class TestProposalGating:
    def test_only_proposed_parcels_can_be_converted(self) -> None:
        env = make_env()
        batch = env.reset(seed=1)
        proposed = (batch.parcels[0],)
        result = env.step(
            np.ones(len(batch.parcels), dtype=np.int64), proposals=proposed
        )
        assert {conversion.parcel for conversion in result.conversions} == set(proposed)

    def test_rejected_proposals_are_not_applied(self) -> None:
        env = make_env()
        batch = env.reset(seed=2)
        result = env.step(
            np.zeros(len(batch.parcels), dtype=np.int64), proposals=(batch.parcels[0],)
        )
        assert result.conversions == ()

    def test_proposals_default_to_the_whole_candidate_set(self) -> None:
        env = make_env(max_conversions_per_iteration=8)
        batch = env.reset(seed=3)
        result = env.step(np.ones(len(batch.parcels), dtype=np.int64))
        assert len(result.conversions) == len(set(batch.parcels))

    def test_proposals_outside_the_candidate_set_are_rejected(self) -> None:
        env = make_env()
        env.reset(seed=4)
        with pytest.raises(ValueError):
            env.step(np.ones(8, dtype=np.int64), proposals=(ParcelIndex(11, 11),))

    def test_proposal_count_is_reported(self) -> None:
        env = make_env()
        batch = env.reset(seed=5)
        result = env.step(
            np.ones(len(batch.parcels), dtype=np.int64),
            proposals=batch.parcels[:2],
        )
        assert result.info["n_proposals"] == pytest.approx(
            float(len(set(batch.parcels[:2])))
        )


class TestConsensusVotes:
    def test_greedy_votes_reproduce_the_published_score_sum(self) -> None:
        scores = [np.array([1.0, -2.0, 0.5]), np.array([0.5, 0.5, -0.25])]
        greedy = [(score > 0).astype(np.int64) for score in scores]
        votes = action_weighted_votes(scores, greedy)
        np.testing.assert_array_equal(
            consensus_decisions(votes), consensus_decisions(scores)
        )

    def test_an_exploratory_flip_reverses_that_stakeholders_vote(self) -> None:
        scores = [np.array([2.0])]
        approve_votes = action_weighted_votes(scores, [np.array([APPROVE])])
        reject_votes = action_weighted_votes(scores, [np.array([REJECT])])
        assert approve_votes[0][0] > 0.0
        assert reject_votes[0][0] < 0.0
        np.testing.assert_array_equal(consensus_decisions(reject_votes), np.array([0]))

    def test_votes_keep_the_magnitude_of_the_quality_score(self) -> None:
        scores = [np.array([-3.0])]
        votes = action_weighted_votes(scores, [np.array([APPROVE])])
        assert votes[0][0] == pytest.approx(3.0)

    def test_rejects_mismatched_action_shapes(self) -> None:
        with pytest.raises(ValueError):
            action_weighted_votes([np.zeros(3)], [np.zeros(2, dtype=np.int64)])

    def test_rejects_a_mismatched_number_of_stakeholders(self) -> None:
        with pytest.raises(ValueError):
            action_weighted_votes([np.zeros(3), np.zeros(3)], [np.zeros(3, np.int64)])


class TestJointTraining:
    def test_runs_and_trains_both_agents(self) -> None:
        env = make_env()
        stakeholders = [make_stakeholder(env)]
        investor = make_investor(env)
        qolu_before = [
            p.detach().clone() for p in stakeholders[0].online_network.parameters()
        ]
        buia_before = [p.detach().clone() for p in investor.network.parameters()]

        report = train_joint(env, stakeholders, investor, iterations=20, seed=0)

        assert isinstance(report, JointTrainingReport)
        assert report.iterations == 20
        assert len(report.team_rewards) == 20
        assert len(report.buia_rewards) == 20
        assert all(np.isfinite(value) for value in report.buia_rewards)
        assert report.qolu_losses and report.buia_losses
        assert any(
            not torch.allclose(a, b)
            for a, b in zip(qolu_before, stakeholders[0].online_network.parameters())
        )
        assert any(
            not torch.allclose(a, b)
            for a, b in zip(buia_before, investor.network.parameters())
        )

    def test_conversions_never_exceed_the_iteration_cap(self) -> None:
        env = make_env(max_conversions_per_iteration=2)
        report = train_joint(
            env, [make_stakeholder(env)], make_investor(env), iterations=10, seed=1
        )
        assert max(report.n_conversions) <= 2

    def test_supports_several_stakeholders(self) -> None:
        env = make_env()
        stakeholders = [make_stakeholder(env, seed=index) for index in range(3)]
        report = train_joint(
            env, stakeholders, make_investor(env), iterations=8, seed=2
        )
        assert len(report.team_rewards) == 8
        assert len(report.qolu_losses) <= 8 * len(stakeholders)

    def test_is_reproducible_for_a_fixed_seed(self) -> None:
        def run() -> JointTrainingReport:
            env = make_env()
            return train_joint(
                env, [make_stakeholder(env)], make_investor(env), iterations=10, seed=7
            )

        np.testing.assert_allclose(run().team_rewards, run().team_rewards, rtol=1e-6)

    def test_reports_well_being_and_metrics(self) -> None:
        env = make_env()
        report = train_joint(
            env, [make_stakeholder(env)], make_investor(env), iterations=8, seed=3
        )
        assert len(report.well_being) == 8
        assert all(0.0 <= value <= 1.0 for value in report.well_being)
        assert set(report.final_metrics) == {
            "land_use",
            "woodland",
            "urban",
            "conversion_alignment",
            "c_align",
            "well_being",
        }

    def test_final_metrics_measure_the_landscape_at_the_end_of_the_run(self) -> None:
        # A run whose length equals the episode length must not be reset before
        # the metrics are taken, or they would describe the initial landscape.
        env = make_env(iterations=5, max_conversions_per_iteration=4)
        report = train_joint(
            env, [make_stakeholder(env)], make_investor(env), iterations=5, seed=8
        )
        assert sum(report.n_conversions) > 0
        assert not np.allclose(env.grid.composition, env.initial_grid.composition)
        assert report.final_metrics["woodland"] <= 1.0
        assert report.urban_fraction[-1] > report.urban_fraction[0]

    def test_qolu_only_runs_still_work_without_an_investor(self) -> None:
        env = make_env()
        report = train_joint(env, [make_stakeholder(env)], None, iterations=6, seed=4)
        assert len(report.team_rewards) == 6
        assert report.buia_losses == ()
        assert all(value == 0.0 for value in report.buia_rewards)

    def test_requires_at_least_one_stakeholder(self) -> None:
        env = make_env()
        with pytest.raises(ValueError):
            train_joint(env, [], make_investor(env), iterations=2, seed=0)


class RecordingQOLUAgent(QOLUAgent):
    """Captures the rewards the runner hands to :meth:`remember`."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.seen_rewards: list[list[float]] = []

    def act(self, observations: np.ndarray, epsilon: float | None = None) -> np.ndarray:
        self.seen_rewards.append([])
        return super().act(observations, epsilon=epsilon)

    def remember(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.seen_rewards[-1].append(float(reward))
        super().remember(observation, action, reward, next_observation, done)


def make_recording_stakeholder(env: UrbanLandUseEnv) -> RecordingQOLUAgent:
    return RecordingQOLUAgent(
        QOLUConfig(
            observation_channels=env.observation_channels,
            observation_window=env.config.observation_window,
            batch_size=8,
            learning_starts=8,
            replay_capacity=200,
            n_step=2,
            epsilon_decay_steps=20,
        ),
        seed=0,
    )


class TestRewardAllocationReadings:
    """Section 3.1 says one shared team reward; Section 3.5 says each QOLU agent
    allocates rewards from its own objectives, per parcel.  All readings run."""

    def test_per_parcel_is_the_default_mode(self) -> None:
        assert QOLURewardConfig().mode == "per_parcel"

    def test_per_parcel_rewards_vary_across_candidates(self) -> None:
        env = make_env()
        agent = make_recording_stakeholder(env)
        train_joint(env, [agent], make_investor(env), iterations=3, seed=5)
        first_iteration = agent.seen_rewards[0]
        assert len(first_iteration) == env.config.samples_per_iteration
        assert len(set(first_iteration)) > 1

    def test_team_mode_gives_every_candidate_the_same_reward(self) -> None:
        env = make_env()
        agent = make_recording_stakeholder(env)
        report = train_joint(
            env,
            [agent],
            make_investor(env),
            iterations=3,
            seed=5,
            qolu_reward=QOLURewardConfig(mode="team"),
        )
        for iteration, rewards in enumerate(agent.seen_rewards):
            assert len(set(rewards)) == 1
            assert rewards[0] == pytest.approx(report.team_rewards[iteration])

    def test_objective_mode_uses_the_stakeholders_own_objective_reward(self) -> None:
        env = make_env()
        agent = make_recording_stakeholder(env)
        train_joint(
            env,
            [agent],
            make_investor(env),
            iterations=3,
            seed=5,
            qolu_reward=QOLURewardConfig(mode="objective"),
            stakeholder_objectives=[TopDownObjective.WOODLAND_PRESERVATION],
        )
        for rewards in agent.seen_rewards:
            assert len(set(rewards)) == 1  # one scalar per iteration

    def test_reward_mode_changes_what_the_agents_learn(self) -> None:
        def run(mode: str) -> QOLUAgent:
            env = make_env()
            stakeholders = [make_stakeholder(env, seed=index) for index in range(3)]
            train_joint(
                env,
                stakeholders,
                make_investor(env),
                iterations=10,
                seed=11,
                qolu_reward=QOLURewardConfig(mode=mode),
            )
            return stakeholders[0]

        shared = run("team")
        per_parcel = run("per_parcel")
        assert any(
            not torch.allclose(a, b)
            for a, b in zip(
                shared.online_network.parameters(),
                per_parcel.online_network.parameters(),
            )
        )

    def test_explicit_objective_assignment_must_match_the_agent_count(self) -> None:
        env = make_env()
        with pytest.raises(ValueError):
            train_joint(
                env,
                [make_stakeholder(env)],
                None,
                iterations=2,
                seed=0,
                stakeholder_objectives=list(TopDownObjective),
            )


class TestApprovalAccounting:
    def test_step_reports_candidates_approvals_proposals_and_conversions(self) -> None:
        env = make_env()
        batch = env.reset(seed=31)
        decisions = np.zeros(len(batch.parcels), dtype=np.int64)
        decisions[:3] = APPROVE
        result = env.step(decisions, proposals=batch.parcels[:1])
        approved_unique = len({batch.parcels[index] for index in range(3)})
        assert result.info["n_candidates"] == pytest.approx(
            float(len(set(batch.parcels)))
        )
        assert result.info["n_approved"] == pytest.approx(float(approved_unique))
        assert result.info["n_conversions"] <= result.info["n_approved"]

    def test_report_carries_the_approval_counters(self) -> None:
        env = make_env()
        report = train_joint(
            env, [make_stakeholder(env)], make_investor(env), iterations=5, seed=32
        )
        assert len(report.n_approved) == 5
        assert len(report.n_candidates) == 5
        assert all(
            conversions <= approved
            for conversions, approved in zip(report.n_conversions, report.n_approved)
        )
