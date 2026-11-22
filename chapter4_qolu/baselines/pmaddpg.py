"""P-MADDPG baseline - the property-based MADDPG of Pelcner et al. (2024).

Published definition (Chapter 3, "Property-based MADDPG" and its Algorithm 1):
a modified MADDPG for *property-based* problems, in which agents are **static**
(they never move) and act on their own cells; actions and rewards are modelled
**per cell**; a multi-objective reward is linearised by a constant; experiences
``{s, a, s', r}`` go into a replay buffer ``Lambda`` that is sampled uniformly;
each agent owns an actor ``upsilon_i`` and a critic ``psi_i`` with target
copies; the critic is trained by MSE against
``targetQ = psi_i^t(s', upsilon_i^t(s')) * gamma + r`` and the actor by
``loss = -mean psi_i(s, upsilon_i(s))``; all of it over per-cell state and
action representations rather than agent-level vectors.

Mapping onto the Chapter 4 environment
--------------------------------------
* One P-MADDPG agent per **top-down stakeholder** (three, one per
  :class:`~chapter4_qolu.objectives.TopDownObjective`).  Stakeholders are static
  and appraise parcels, which is exactly the property-based setting.
* Every agent appraises **every candidate parcel** through the same observation
  window the QOLU agents use, so the two methods see identical inputs.
* The continuous actor output in [0, 1] is an *approve-score*, thresholded at
  ``config.approve_threshold`` (0.5) into the published ``{REJECT, APPROVE}``
  action set.  Ornstein-Uhlenbeck noise explores the score, not the bit.
* The per-cell reward is the **same** per-parcel compliance reward the QOLU
  stakeholders learn from (:func:`~chapter4_qolu.env.qolu_rewards
  .qolu_parcel_rewards` with :class:`~chapter4_qolu.config.QOLURewardConfig`),
  so a difference in results is a difference in *algorithm*, not in reward.
* Decisions are combined with the published consensus rule
  (:mod:`chapter4_qolu.agent.consensus`), so the environment step is identical
  to the one ``train_joint`` performs.
* Because both of those are shared, so are the two optional routes by which the
  scenario weights reach a top-down agent
  (:attr:`~chapter4_qolu.config.QOLURewardConfig.scale_by_scenario_weight` and
  :attr:`~chapter4_qolu.config.QOLURewardConfig.weighted_consensus`): a campaign
  that turns one on weights P-MADDPG exactly as it weights QOLU + BUIA.

Deliberately **not** ported: Chapter 3's four-policy curriculum (Policies #0-#3).
Those policies are defined in terms of pollinator counts and neighbouring
habitat in the Pollinators Game and have no counterpart in a land-conversion
appraisal, so importing them would invent behaviour rather than reproduce it.

Hyper-parameters and the actor loss
-----------------------------------
Chapter 3 publishes optimisation hyper-parameters (lines 392-393:
``gamma = 0.1``, a hard target refresh "every 250 steps", ``epsilon = 0.3`` and
a learning rate of 1e-4), and those are the defaults of
:class:`~chapter4_qolu.baselines.config.PMADDPGConfig`.  The actor loss adds the
policy regulariser Lowe et al.'s MADDPG implementation uses, which keeps the
pre-activation in range: without it the sigmoid saturates within a handful of
updates and the run locks into an approve-all or reject-all corner.  The
unregularised reading stays reachable as ``variant="v1"``; see
:mod:`chapter4_qolu.baselines.pmaddpg_variants`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np
import torch

from chapter4_qolu.baselines.base import BaselineDecision, BaselineMethod
from chapter4_qolu.baselines.config import PMADDPGConfig
from chapter4_qolu.baselines.noise import OrnsteinUhlenbeckNoise
from chapter4_qolu.baselines.pmaddpg_agent import PMADDPGAgent, target_q_value
from chapter4_qolu.baselines.pmaddpg_decisions import (
    appraisal_consensus,
    epsilon_greedy_scores,
    ornstein_uhlenbeck_scores,
    resized_noise,
)
from chapter4_qolu.baselines.pmaddpg_networks import (
    JointReplayBuffer,
    JointTransition,
    JointTransitionBatch,
)
from chapter4_qolu.baselines.pmaddpg_variants import (
    PMADDPG_VARIANTS,
    SeedStreams,
    seed_streams,
    variant_settings,
)
from chapter4_qolu.config import QOLURewardConfig
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.env.qolu_rewards import qolu_parcel_rewards
from chapter4_qolu.env.urban_env import DecisionBatch, StepResult, UrbanLandUseEnv
from chapter4_qolu.objectives import TopDownObjective

__all__ = [
    "PMADDPG_FACTORIES",
    "PMADDPG_VARIANTS",
    "PMADDPGAgent",
    "PMADDPGMethod",
    "appraisal_consensus",
    "pmaddpg_reg_only",
    "pmaddpg_v1",
    "published_pmaddpg",
    "target_q_value",
    "variant_settings",
]


class PMADDPGMethod(BaselineMethod):
    """The P-MADDPG baseline as a pluggable :class:`BaselineMethod`."""

    def __init__(self, config: PMADDPGConfig, seed: int | None = None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.seeds: SeedStreams = seed_streams(seed, config.n_agents)
        self.rng = self.seeds.generator("replay")
        self.agents = [
            PMADDPGAgent(config, index, agent_seed)
            for index, agent_seed in enumerate(self.seeds.agent_seeds("initialisation"))
        ]
        self.replay = JointReplayBuffer(config.replay_capacity, config.n_agents)
        self.exploration = self.seeds.generator("noise")
        #: One process per agent, re-sized to the decision buffer on first use.
        self.noise = [
            OrnsteinUhlenbeckNoise(
                size=1,
                theta=config.ou_theta,
                sigma=config.ou_sigma,
                dt=config.ou_dt,
                rng=rng,
            )
            for rng in self.seeds.agent_generators("noise")
        ]
        self.noise_scale = 1.0
        #: The reward reading the agents are actually paid, i.e. ``qolu_reward``
        #: with Chapter 3's linearisation constant folded in (see
        #: :attr:`~chapter4_qolu.baselines.config.PMADDPGConfig
        #: .team_reward_weight`).  Resolved once, here, so that a campaign which
        #: hands the method its own :class:`QOLURewardConfig` cannot erase it.
        self.qolu_reward = _with_linearisation(
            config.qolu_reward, config.team_reward_weight
        )
        #: Design note: agents are assigned objectives in declaration order and
        #: wrap around, exactly as ``runner._stakeholder_objectives`` does.
        available = list(TopDownObjective)
        self.objectives = tuple(
            available[index % len(available)] for index in range(config.n_agents)
        )

        self._env: UrbanLandUseEnv | None = None
        self._grid_before: ParcelGrid | None = None
        self._batch: DecisionBatch | None = None
        self.last_actions: np.ndarray | None = None
        self.last_rewards: np.ndarray | None = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def for_env(
        cls,
        env: UrbanLandUseEnv,
        seed: int | None = None,
        variant: str = "published",
        **overrides: object,
    ) -> PMADDPGMethod:
        """Build a method whose observation shapes match ``env``.

        ``variant`` names one of the three readings in
        :data:`~chapter4_qolu.baselines.pmaddpg_variants.PMADDPG_VARIANTS`;
        ``overrides`` are applied *after* it, so an explicit keyword always
        wins over the variant it belongs to.
        """
        settings: dict[str, object] = {
            "observation_channels": env.observation_channels,
            "observation_window": env.config.observation_window,
        }
        settings.update(variant_settings(variant))
        settings.update(overrides)
        return cls(PMADDPGConfig(**settings), seed=seed)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        return "p_maddpg"

    # ------------------------------------------------------------------
    # acting
    # ------------------------------------------------------------------
    def reset(self, env: UrbanLandUseEnv, seed: int | None = None) -> None:
        if env.observation_channels != self.config.observation_channels:
            raise ValueError(
                f"environment supplies {env.observation_channels} observation "
                f"channels, config expects {self.config.observation_channels}"
            )
        self._env = env
        self._grid_before = None
        self._batch = None
        streams = seed_streams(seed, self.config.n_agents)
        self.exploration = streams.generator("reset")
        for noise, rng in zip(self.noise, streams.agent_generators("reset")):
            noise.reset(rng)

    def approve_scores(self, observations: np.ndarray) -> np.ndarray:
        """Every agent's noiseless approve-score, shape ``(n_agents, batch)``."""
        tensor = self._to_tensor(observations)
        return np.stack([agent.approve_scores(tensor) for agent in self.agents])

    def decide(self, batch: DecisionBatch) -> BaselineDecision:
        if self._env is None:
            raise RuntimeError("call reset() before decide()")
        self._grid_before = self._env.grid
        self._batch = batch

        tensor = self._to_tensor(batch.observations)
        scores = [
            np.clip(self._explore(agent.approve_scores(tensor), index), 0.0, 1.0)
            for index, agent in enumerate(self.agents)
        ]

        self.last_actions = np.stack(scores, axis=1).astype(np.float32)
        decisions = appraisal_consensus(
            scores, self.config.approve_threshold, weights=self._consensus_weights()
        )
        return BaselineDecision(decisions=decisions)

    # ------------------------------------------------------------------
    # exploration
    # ------------------------------------------------------------------
    def _explore(self, values: np.ndarray, index: int) -> np.ndarray:
        """Chapter 3's epsilon-greedy, or DDPG's OU noise when it is ``None``."""
        epsilon = self.config.epsilon_greedy
        if epsilon is not None:
            return epsilon_greedy_scores(values, float(epsilon), self.exploration)
        if self.config.ou_sigma <= 0.0:
            return values
        noise = resized_noise(self.noise[index], int(values.size), self.config)
        self.noise[index] = noise
        return ornstein_uhlenbeck_scores(values, noise, self.noise_scale)

    def _consensus_weights(self) -> tuple[float, ...] | None:
        """The scenario weights of the run, per agent, or ``None``.

        Design note: the optional weighted-consensus reading is taken from the
        *environment's* weighting rather than from a copy held by the method,
        so the baseline and ``train_joint`` can never be weighted differently
        inside the same campaign cell.  ``None`` - the default - is the
        published plain sum.
        """
        if not self.qolu_reward.weighted_consensus:
            return None
        assert self._env is not None
        return tuple(self._env.weights[objective] for objective in self.objectives)

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def observe(self, result: StepResult) -> None:
        if self._grid_before is None or self._batch is None:
            raise RuntimeError("call decide() before observe()")
        actions = self.last_actions
        assert actions is not None

        rewards = np.stack(
            [
                self._objective_rewards(result, actions[:, index], objective)
                for index, objective in enumerate(self.objectives)
            ],
            axis=1,
        )
        self.last_rewards = rewards

        for position in range(len(result.parcels)):
            self.remember(
                JointTransition(
                    observation=self._batch.observations[position],
                    actions=actions[position],
                    rewards=rewards[position].astype(np.float32),
                    next_observation=result.next_observations[position],
                    done=result.done,
                )
            )

        self.noise_scale = max(
            self.config.min_noise_scale, self.noise_scale * self.config.noise_decay
        )

    def _objective_rewards(
        self, result: StepResult, scores: np.ndarray, objective: TopDownObjective
    ) -> np.ndarray:
        """One stakeholder's per-cell reward for this synchronised update.

        The three readings of ``R_QOLU,i`` are the same ones
        :func:`chapter4_qolu.runner.train_joint` offers, so a baseline run and a
        QOLU + BUIA run can be compared under whichever reading is being
        reported.  The default is the per-parcel compliance signal.
        """
        assert self._env is not None and self._grid_before is not None
        mode = self.qolu_reward.mode
        if mode == "per_parcel":
            taken = (scores >= self.config.approve_threshold).astype(np.int64)
            return qolu_parcel_rewards(
                self._grid_before,
                result.parcels,
                taken,
                objective,
                # Design note: P-MADDPG has no bottom-up investor, so every
                # candidate counts as proposed - the same reading the
                # environment uses for ``proposals=None``.
                proposed=None,
                config=self._env.config.reward,
                qolu_reward=self.qolu_reward,
                team_reward=result.team_reward,
                # Same source as ``train_joint``: the environment's own
                # weighting, used only by ``scale_by_scenario_weight``.
                scenario_weights=self._env.weights,
                n_stakeholders=self.config.n_agents,
            )
        scalar = (
            float(result.objective_rewards[objective])
            if mode == "objective"
            else float(result.team_reward)
        )
        return np.full(len(result.parcels), scalar, dtype=np.float64)

    def remember(self, transition: JointTransition) -> None:
        """Store one cell's ``{s, a, s', r}`` in ``Lambda``."""
        self.replay.push(transition)

    def train_step(self) -> float | None:
        """One pass of Chapter 3's Algorithm 1 over all agents."""
        ready = max(self.config.learning_starts, self.config.batch_size)
        if len(self.replay) < ready:
            return None

        losses = []
        for index, agent in enumerate(self.agents):
            # "delta ~ Lambda" sits *inside* the per-agent loop of Algorithm 1,
            # so every agent draws its own uniform mini-batch.
            batch = self.replay.sample(self.config.batch_size, self.rng)
            tensors = self._batch_tensors(batch)
            # Design note (Algorithm 1 line 6): Chapter 3 writes the target
            # action as ``a'_i = upsilon^t_i(s'_i)`` - agent i's target actor
            # alone.  The critic here is MADDPG's *centralised* one, so it has
            # to be fed an ``n_agents``-vector; the other slots are therefore
            # filled from the other agents' target actors, which is standard
            # MADDPG and the only reading under which the target critic sees
            # the same kind of input as the online critic.  Chapter 3's literal
            # reading - agent i's action in every slot - would make the three
            # critics see joint actions that no policy can produce.
            with torch.no_grad():
                next_joint = torch.cat(
                    [
                        other.actor_target(tensors["next_observations"])
                        for other in self.agents
                    ],
                    dim=1,
                )
            losses.append(
                agent.update_critic(
                    tensors["observations"],
                    tensors["actions"],
                    tensors["rewards"][:, index],
                    tensors["next_observations"],
                    next_joint,
                    tensors["dones"],
                )
            )
            agent.update_actor(tensors["observations"], tensors["actions"], index)
            agent.update_targets()
        return float(np.mean(losses))

    # ------------------------------------------------------------------
    def _batch_tensors(self, batch: JointTransitionBatch) -> dict[str, torch.Tensor]:
        return {
            "observations": self._to_tensor(batch.observations),
            "next_observations": self._to_tensor(batch.next_observations),
            "actions": self._to_tensor(batch.actions),
            "rewards": self._to_tensor(batch.rewards),
            "dones": self._to_tensor(batch.dones),
        }

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(array, dtype=np.float32), dtype=torch.float32, device=self.device
        )


def _with_linearisation(
    qolu_reward: QOLURewardConfig, team_reward_weight: float | None
) -> QOLURewardConfig:
    """``qolu_reward`` with Chapter 3's linearisation constant folded in.

    ``None`` leaves the supplied reading exactly as it is, which is what a
    campaign that wants to drive the weight from ``QOLURewardConfig`` itself
    should use.  Otherwise the weight *overrides* whatever the caller supplied,
    because :func:`chapter4_qolu.scenario_worker._build_baseline` hands every
    P-MADDPG run the campaign's own :class:`QOLURewardConfig` - in which
    ``team_reward_weight`` is the published-QOLU default of 0.0 - and a
    P-MADDPG-only constant has to survive that.
    """
    if team_reward_weight is None:
        return qolu_reward
    return replace(qolu_reward, team_reward_weight=float(team_reward_weight))


def _variant_factory(variant: str) -> Callable[..., PMADDPGMethod]:
    """A named constructor for one reading, mirroring ``fixed_ltos``."""

    def build(
        env: UrbanLandUseEnv, seed: int | None = None, **overrides: object
    ) -> PMADDPGMethod:
        return PMADDPGMethod.for_env(env, seed=seed, variant=variant, **overrides)

    build.__name__ = f"pmaddpg_{variant}"
    build.__qualname__ = build.__name__
    build.__doc__ = (
        f"P-MADDPG under the {variant!r} reading; see "
        "``chapter4_qolu.baselines.pmaddpg_variants``."
    )
    return build


#: Chapter 3's published hyper-parameters plus the policy regulariser.
published_pmaddpg = _variant_factory("published")
#: Adam 1e-3, gamma 0.99, Polyak, OU, no regulariser.
pmaddpg_v1 = _variant_factory("v1")
#: ``v1`` plus the regulariser, and nothing else.
pmaddpg_reg_only = _variant_factory("reg_only")

#: Variant name -> its named constructor, for callers that would rather map
#: than pass a string (``scenario_worker`` passes the string).
PMADDPG_FACTORIES: Mapping[str, object] = {
    "published": published_pmaddpg,
    "v1": pmaddpg_v1,
    "reg_only": pmaddpg_reg_only,
}
