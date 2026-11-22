"""Training loops for the Chapter 4 agents.

* :func:`train_qolu` is Algorithm 1's actor/learner loop with a single actor and
  no investor: the top-down agent appraises every sampled candidate.
* :func:`train_joint` is the full Section 3.5 iteration - BUIA proposes, the
  QOLU stakeholders appraise, the approved changes are applied synchronously,
  and both sides learn from their own reward.

Neither is the paper's four-scenario experiment runner: that is
:mod:`chapter4_qolu.run_scenarios`, which drives both of these entry points
across every method, scenario and seed.  The three published baselines have
their own, parallel entry point in
:func:`chapter4_qolu.baselines.base.run_baseline`, deliberately kept separate so
that the baselines cannot change QOLU + BUIA behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chapter4_qolu.agent.buia_agent import BUIAAgent
from chapter4_qolu.agent.checkpoint import save_agents
from chapter4_qolu.agent.consensus import action_weighted_votes, consensus_decisions
from chapter4_qolu.agent.qolu_agent import QOLUAgent
from chapter4_qolu.config import QOLURewardConfig
from chapter4_qolu.env.evaluation import evaluate_landscape
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.env.qolu_rewards import qolu_parcel_rewards
from chapter4_qolu.env.urban_env import StepResult, UrbanLandUseEnv
from chapter4_qolu.land_cover import LandCoverGroup
from chapter4_qolu.objectives import TopDownObjective

__all__ = ["JointTrainingReport", "TrainingReport", "train_joint", "train_qolu"]


def _learn_steps(agent: QOLUAgent) -> list[float]:
    """Fire one iteration's gradient budget, returning the losses it produced.

    ``QOLUConfig.learn_steps_per_iteration`` defaults to 1 - a single
    ``learn()`` call per iteration in both loops below, which is Algorithm 1
    read literally.  A larger budget repeats the "sample a minibatch, take a gradient step" half
    of the algorithm without touching the actor half, so the environment, the
    replay contents and the reward are unchanged and only the number of times
    each stored transition is revisited moves.

    ``learn()`` returns ``None`` only while the buffer is below
    ``learning_starts``, and nothing is pushed between the calls here, so the
    first ``None`` settles the whole budget for this iteration.
    """
    losses: list[float] = []
    for _ in range(agent.config.learn_steps_per_iteration):
        loss = agent.learn()
        if loss is None:
            break
        losses.append(loss)
    return losses


@dataclass(frozen=True)
class TrainingReport:
    """Per-iteration diagnostics of a training run."""

    iterations: int
    episodes: int
    team_rewards: tuple[float, ...]
    losses: tuple[float, ...]
    urban_fraction: tuple[float, ...]
    conversion_alignment: tuple[float, ...]
    epsilons: tuple[float, ...]

    @property
    def mean_team_reward(self) -> float:
        return float(np.mean(self.team_rewards)) if self.team_rewards else 0.0


def train_qolu(
    env: UrbanLandUseEnv,
    agent: QOLUAgent,
    iterations: int,
    seed: int | None = None,
    epsilon: float | None = None,
) -> TrainingReport:
    """Run ``iterations`` synchronised environment updates, learning throughout.

    Every candidate appraisal in an iteration becomes its own transition and all
    of them receive the *same* shared team reward, which is how the paper keeps
    the setting fully cooperative.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")

    batch = env.reset(seed=seed)
    episodes = 1
    team_rewards: list[float] = []
    losses: list[float] = []
    urban_fraction: list[float] = []
    alignment: list[float] = []
    epsilons: list[float] = []

    for _ in range(iterations):
        rate = agent.epsilon() if epsilon is None else float(epsilon)
        epsilons.append(rate)
        actions = agent.act(batch.observations, epsilon=rate)
        result = env.step(actions)

        for index in range(len(result.parcels)):
            agent.remember(
                batch.observations[index],
                int(actions[index]),
                result.team_reward,
                result.next_observations[index],
                result.done,
            )

        losses.extend(_learn_steps(agent))

        team_rewards.append(result.team_reward)
        urban_fraction.append(
            float(env.grid.group_fraction(LandCoverGroup.URBAN).mean())
        )
        alignment.append(float(result.info["conversion_alignment"]))

        if result.done:
            episodes += 1
            batch = env.reset(seed=None if seed is None else seed + episodes)
        else:
            batch = result.next_batch

    return TrainingReport(
        iterations=iterations,
        episodes=episodes,
        team_rewards=tuple(team_rewards),
        losses=tuple(losses),
        urban_fraction=tuple(urban_fraction),
        conversion_alignment=tuple(alignment),
        epsilons=tuple(epsilons),
    )


def _transition_rewards(
    grid_before: ParcelGrid,
    parcels: Sequence[ParcelIndex],
    actions: np.ndarray,
    objective: TopDownObjective,
    proposed: Sequence[ParcelIndex] | None,
    result: StepResult,
    env: UrbanLandUseEnv,
    qolu_reward: QOLURewardConfig,
    n_stakeholders: int,
) -> np.ndarray:
    """The reward each appraisal in this iteration earns for one stakeholder."""
    if qolu_reward.mode == "per_parcel":
        return qolu_parcel_rewards(
            grid_before,
            parcels,
            actions,
            objective,
            proposed=None if proposed is None else set(proposed),
            config=env.config.reward,
            qolu_reward=qolu_reward,
            team_reward=result.team_reward,
            # The run's preference weighting, used only by
            # ``qolu_reward.scale_by_scenario_weight``.
            scenario_weights=env.weights,
            n_stakeholders=n_stakeholders,
        )
    scalar = (
        float(result.objective_rewards[objective])
        if qolu_reward.mode == "objective"
        else float(result.team_reward)
    )
    return np.full(len(parcels), scalar, dtype=np.float64)


def _stakeholder_objectives(
    stakeholders: Sequence[QOLUAgent],
    objectives: Sequence[TopDownObjective] | None,
) -> tuple[TopDownObjective, ...]:
    """Map each stakeholder onto the objective it represents.

    Design note: the paper runs three top-down stakeholders, one per published
    preference, so stakeholders are assigned objectives in declaration order and
    wrap around if more agents than objectives are supplied.
    """
    if objectives is None:
        available = list(TopDownObjective)
        return tuple(
            available[index % len(available)] for index in range(len(stakeholders))
        )
    objectives = tuple(objectives)
    if len(objectives) != len(stakeholders):
        raise ValueError(
            f"got {len(objectives)} objectives for {len(stakeholders)} stakeholders"
        )
    return objectives


@dataclass(frozen=True)
class JointTrainingReport:
    """Per-iteration diagnostics of a joint BUIA + QOLU run."""

    iterations: int
    episodes: int
    team_rewards: tuple[float, ...]
    buia_rewards: tuple[float, ...]
    qolu_losses: tuple[float, ...]
    buia_losses: tuple[float, ...]
    urban_fraction: tuple[float, ...]
    conversion_alignment: tuple[float, ...]
    well_being: tuple[float, ...]
    n_conversions: tuple[int, ...]
    n_proposals: tuple[int, ...]
    n_approved: tuple[int, ...]
    n_candidates: tuple[int, ...]
    epsilons: tuple[float, ...]
    final_metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def mean_team_reward(self) -> float:
        return float(np.mean(self.team_rewards)) if self.team_rewards else 0.0

    @property
    def mean_buia_reward(self) -> float:
        return float(np.mean(self.buia_rewards)) if self.buia_rewards else 0.0


def train_joint(
    env: UrbanLandUseEnv,
    stakeholders: Sequence[QOLUAgent],
    investor: BUIAAgent | None,
    iterations: int,
    seed: int | None = None,
    epsilon: float | None = None,
    qolu_reward: QOLURewardConfig | None = None,
    stakeholder_objectives: Sequence[TopDownObjective] | None = None,
    checkpoint_dir: Path | str | None = None,
) -> JointTrainingReport:
    """Run the paper's Section 3.5 iteration for ``iterations`` steps.

    Order of play, following Section 3.5 and Experimental Settings (ii)-(iii):
    the decision buffer is sampled proportionally to attractiveness, the
    investor proposes a subset of it, the stakeholders appraise every candidate,
    and the approved proposals are applied in one synchronised update.

    Design note (gating): a proposal is applied when the stakeholders' summed
    quality score is positive - the paper's consensus rule.  During training the
    sign of each stakeholder's contribution follows the epsilon-greedy action it
    actually took, so exploration still moves the environment; see
    :func:`chapter4_qolu.agent.consensus.action_weighted_votes`.

    Design note (credit): every stakeholder stores its *own* action against the
    shared team reward, which is the paper's fully cooperative setting; with a
    single stakeholder and greedy actions the applied decision is exactly that
    agent's decision.

    ``qolu_reward`` selects between the paper's contradictory reward
    allocations: ``"per_parcel"`` (default, Section 3.5's
    ``R_QOLU,i = f(parcel_data, objectives_i)`` evaluated for each appraised
    parcel), ``"objective"`` (Section 3.5 at iteration level) and ``"team"``
    (Section 3.1's shared reward).  See
    :class:`chapter4_qolu.config.QOLURewardConfig`.

    Design note (scenario weights, both off by default): ``env.weights`` reaches
    the stakeholders only when ``qolu_reward`` asks for it - as a per-parcel
    reward scale (``scale_by_scenario_weight``) and/or as a weight on each
    stakeholder's contribution to the consensus (``weighted_consensus``).  With
    both off, ``w_i`` affects nothing but the reported team reward, which is
    the published reading.

    ``checkpoint_dir`` (off by default) writes each stakeholder to
    ``<dir>/stakeholder<i>.pt`` once the run has finished, so the trained
    networks outlive the run and the return distributions they learnt can be
    scored afterwards - see :mod:`chapter4_qolu.agent.checkpoint` and
    :mod:`chapter4_qolu.diagnostics`.  It is written *after* the final metrics
    are evaluated, so a checkpoint on disk means the run completed.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    stakeholders = list(stakeholders)
    if not stakeholders:
        raise ValueError("at least one QOLU stakeholder is required")
    objectives = _stakeholder_objectives(stakeholders, stakeholder_objectives)
    qolu_reward = qolu_reward or QOLURewardConfig()
    consensus_weights = (
        tuple(env.weights[objective] for objective in objectives)
        if qolu_reward.weighted_consensus
        else None
    )

    batch = env.reset(seed=seed)
    episodes = 1
    team_rewards: list[float] = []
    buia_rewards: list[float] = []
    qolu_losses: list[float] = []
    buia_losses: list[float] = []
    urban_fraction: list[float] = []
    alignment: list[float] = []
    well_being: list[float] = []
    n_conversions: list[int] = []
    n_proposals: list[int] = []
    n_approved: list[int] = []
    n_candidates: list[int] = []
    epsilons: list[float] = []

    for iteration in range(iterations):
        rate = stakeholders[0].epsilon() if epsilon is None else float(epsilon)
        epsilons.append(rate)

        proposal_parcels = None
        if investor is not None:
            unique_candidates = tuple(dict.fromkeys(batch.parcels))
            proposal = investor.propose(
                unique_candidates,
                env.buia_observations(unique_candidates),
                max_proposals=env.config.max_conversions_per_iteration,
            )
            proposal_parcels = proposal.parcels

        actions = [
            agent.act(batch.observations, epsilon=rate) for agent in stakeholders
        ]
        scores = [agent.quality_scores(batch.observations) for agent in stakeholders]
        decisions = consensus_decisions(
            action_weighted_votes(scores, actions, weights=consensus_weights)
        )

        grid_before = env.grid  # compliance is judged on the appraised landscape
        result = env.step(decisions, proposals=proposal_parcels)

        for agent, agent_actions, objective in zip(stakeholders, actions, objectives):
            rewards = _transition_rewards(
                grid_before=grid_before,
                parcels=result.parcels,
                actions=agent_actions,
                objective=objective,
                proposed=proposal_parcels,
                result=result,
                env=env,
                qolu_reward=qolu_reward,
                n_stakeholders=len(stakeholders),
            )
            for index in range(len(result.parcels)):
                agent.remember(
                    batch.observations[index],
                    int(agent_actions[index]),
                    float(rewards[index]),
                    result.next_observations[index],
                    result.done,
                )
            qolu_losses.extend(_learn_steps(agent))

        if investor is not None:
            reward = env.attractiveness_delta(
                result.conversions, mode=investor.config.reward_mode
            )
            buia_rewards.append(reward)
            buia_losses.append(investor.update(reward))
        else:
            buia_rewards.append(0.0)

        team_rewards.append(result.team_reward)
        urban_fraction.append(
            float(env.grid.group_fraction(LandCoverGroup.URBAN).mean())
        )
        alignment.append(float(result.info["conversion_alignment"]))
        well_being.append(float(result.info["well_being"]))
        n_conversions.append(len(result.conversions))
        n_proposals.append(int(result.info["n_proposals"]))
        n_approved.append(int(result.info["n_approved"]))
        n_candidates.append(int(result.info["n_candidates"]))

        if result.done and iteration < iterations - 1:
            # Never reset on the last iteration: the final landscape is what the
            # evaluation metrics below have to measure.
            episodes += 1
            batch = env.reset(seed=None if seed is None else seed + episodes)
        else:
            batch = result.next_batch

    final_metrics = evaluate_landscape(
        env.initial_grid,
        env.grid,
        attractiveness=env.attractiveness,
        population=env.population,
        reward_config=env.config.reward,
        well_being_config=env.config.well_being,
    )
    if checkpoint_dir is not None:
        save_agents(stakeholders, checkpoint_dir)

    return JointTrainingReport(
        iterations=iterations,
        episodes=episodes,
        team_rewards=tuple(team_rewards),
        buia_rewards=tuple(buia_rewards),
        qolu_losses=tuple(qolu_losses),
        buia_losses=tuple(buia_losses),
        urban_fraction=tuple(urban_fraction),
        conversion_alignment=tuple(alignment),
        well_being=tuple(well_being),
        n_conversions=tuple(n_conversions),
        n_proposals=tuple(n_proposals),
        n_approved=tuple(n_approved),
        n_candidates=tuple(n_candidates),
        epsilons=tuple(epsilons),
        final_metrics=final_metrics,
    )
