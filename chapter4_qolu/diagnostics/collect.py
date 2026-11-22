"""Run a trained QOLU stakeholder set forward and log what it predicted.

The rollout is deliberately a *frozen* copy of the Section 3.5 iteration in
:func:`chapter4_qolu.runner.train_joint`: the same decision buffer, the same
investor proposal, the same consensus rule and - importantly - the same reward
function, imported from the runner rather than copied, so the realised return a
prediction is scored against is the quantity the agent was actually paid.

What is different, and why:

* **no learning.**  Nothing is pushed into the replay buffer and no gradient
  step is taken, so the network scored is the network that was loaded.
* **no exploration.**  ``epsilon = 0`` and the NoisyNet noise is zeroed, so
  actions are ``argmax_a E[Z(s, a)]`` on the mean parameters - the same reading
  :meth:`~chapter4_qolu.agent.qolu_agent.QOLUAgent.quality_scores` uses and
  therefore the reading the consensus actually acts on.
* **the investor does not update.**  Its proposals are a property of the
  environment during the evaluation, not something being trained.
* **the environment is reset first.**  A diagnostic measures the agent, so it
  starts from the landscape the run started from rather than from wherever the
  training run happened to leave it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from chapter4_qolu.agent.buia_agent import BUIAAgent
from chapter4_qolu.agent.consensus import action_weighted_votes, consensus_decisions
from chapter4_qolu.agent.qolu_agent import QOLUAgent
from chapter4_qolu.config import QOLURewardConfig
from chapter4_qolu.diagnostics.records import AppraisalLog, DiagnosticsLog
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.objectives import TopDownObjective

# Imported rather than reimplemented on purpose: the diagnostic has to pay the
# stakeholder exactly what training paid it, or the "realised return" it scores
# against is a different quantity from the one the network was fitted to.  A
# local copy of either helper would drift the first time the runner changed.
from chapter4_qolu.runner import _stakeholder_objectives, _transition_rewards

__all__ = ["collect_appraisals"]


@torch.no_grad()
def _bootstrap_values(agent: QOLUAgent, next_observations: np.ndarray) -> np.ndarray:
    """``Q(s', a*)`` with ``a*`` chosen online and evaluated on ``theta^-``.

    Exactly the double-Q rule :meth:`QOLUAgent.learn` bootstraps with, so the
    ``n_step_target`` reading of the realised return is the target the network
    was regressed onto and not an approximation of it.
    """
    selected = agent.q_values(next_observations).argmax(axis=1)
    tensor = torch.as_tensor(
        np.asarray(next_observations, dtype=np.float32),
        dtype=torch.float32,
        device=agent.device,
    )
    agent.target_network.zero_noise()
    values = agent.target_network.q_values(tensor).cpu().numpy()
    return values[np.arange(values.shape[0]), selected]


def _chosen_probabilities(
    agent: QOLUAgent, observations: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    """``p(s, a)`` for the action actually taken, one row per appraisal."""
    distributions = agent.action_distributions(observations)
    return distributions[np.arange(distributions.shape[0]), actions]


def collect_appraisals(
    env: UrbanLandUseEnv,
    stakeholders: Sequence[QOLUAgent],
    investor: BUIAAgent | None = None,
    *,
    iterations: int,
    seed: int | None = 0,
    qolu_reward: QOLURewardConfig | None = None,
    stakeholder_objectives: Sequence[TopDownObjective] | None = None,
    reference_seed: int = 0,
) -> DiagnosticsLog:
    """Appraise ``iterations`` decision buffers and log every prediction made.

    ``reference_seed`` seeds the untrained copies of each network that supply
    the "at initialisation" CRPS reference; they are built from the trained
    agent's own :class:`~chapter4_qolu.config.QOLUConfig`, so the comparison is
    like for like in architecture, support and atom count.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    agents = list(stakeholders)
    if not agents:
        raise ValueError("at least one QOLU stakeholder is required")
    objectives = _stakeholder_objectives(agents, stakeholder_objectives)
    reward_config = qolu_reward or QOLURewardConfig()
    consensus_weights = (
        tuple(env.weights[objective] for objective in objectives)
        if reward_config.weighted_consensus
        else None
    )
    references = [
        QOLUAgent(agent.config, seed=reference_seed + index)
        for index, agent in enumerate(agents)
    ]

    keys = (
        "probabilities",
        "reference",
        "actions",
        "quality",
        "rewards",
        "bootstrap",
        "iteration",
    )
    columns: list[dict[str, list[np.ndarray]]] = [
        {key: [] for key in keys} for _ in agents
    ]

    batch = env.reset(seed=seed)
    episodes = 1
    for iteration in range(iterations):
        proposal_parcels = None
        if investor is not None:
            unique_candidates = tuple(dict.fromkeys(batch.parcels))
            proposal_parcels = investor.propose(
                unique_candidates,
                env.buia_observations(unique_candidates),
                max_proposals=env.config.max_conversions_per_iteration,
            ).parcels

        scores = [agent.quality_scores(batch.observations) for agent in agents]
        actions = [
            agent.q_values(batch.observations).argmax(axis=1).astype(np.int64)
            for agent in agents
        ]
        predicted = [
            _chosen_probabilities(agent, batch.observations, taken)
            for agent, taken in zip(agents, actions)
        ]
        untrained = [
            _chosen_probabilities(reference, batch.observations, taken)
            for reference, taken in zip(references, actions)
        ]
        decisions = consensus_decisions(
            action_weighted_votes(scores, actions, weights=consensus_weights)
        )

        grid_before = env.grid
        result = env.step(decisions, proposals=proposal_parcels)
        count = len(result.parcels)

        for index, (agent, taken, objective) in enumerate(
            zip(agents, actions, objectives)
        ):
            rewards = _transition_rewards(
                grid_before=grid_before,
                parcels=result.parcels,
                actions=taken,
                objective=objective,
                proposed=proposal_parcels,
                result=result,
                env=env,
                qolu_reward=reward_config,
                n_stakeholders=len(agents),
            )
            store = columns[index]
            store["probabilities"].append(predicted[index])
            store["reference"].append(untrained[index])
            store["actions"].append(taken)
            # The very scores the consensus was formed from, kept rather than
            # discarded: their spread is what separates a state-dependent
            # policy from a constant one with the same approval rate.
            store["quality"].append(np.asarray(scores[index], dtype=np.float64))
            store["rewards"].append(np.asarray(rewards, dtype=np.float64))
            store["bootstrap"].append(
                _bootstrap_values(agent, result.next_observations)
            )
            store["iteration"].append(np.full(count, iteration, dtype=np.int64))

        if result.done and iteration < iterations - 1:
            episodes += 1
            batch = env.reset(seed=None if seed is None else seed + episodes)
        else:
            batch = result.next_batch

    logs = tuple(
        AppraisalLog(
            stakeholder=index,
            objective=objective.value,
            atoms=agent.support.atoms().cpu().numpy().astype(np.float64),
            probabilities=np.concatenate(columns[index]["probabilities"]).astype(
                np.float64
            ),
            reference_probabilities=np.concatenate(columns[index]["reference"]).astype(
                np.float64
            ),
            actions=np.concatenate(columns[index]["actions"]),
            quality_scores=np.concatenate(columns[index]["quality"]).astype(np.float64),
            rewards=np.concatenate(columns[index]["rewards"]),
            bootstrap_values=np.concatenate(columns[index]["bootstrap"]).astype(
                np.float64
            ),
            iteration_index=np.concatenate(columns[index]["iteration"]),
            gamma=float(agent.config.gamma),
            n_step=int(agent.config.n_step),
        )
        for index, (agent, objective) in enumerate(zip(agents, objectives))
    )
    return DiagnosticsLog(
        logs=logs,
        iterations=int(iterations),
        samples_per_iteration=int(env.config.samples_per_iteration),
        qolu_reward_mode=reward_config.mode,
        scenario_weights=tuple(
            float(env.weights[objective]) for objective in TopDownObjective
        ),
        epsilon=0.0,
        investor=investor is not None,
        notes=(
            "greedy on the mean parameters (epsilon = 0, NoisyNet noise zeroed)",
            "no gradient step and no replay push during the rollout",
            f"{episodes} episode(s) inside the rollout",
        ),
    )
