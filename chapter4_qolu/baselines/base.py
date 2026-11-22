"""Common interface and driver for the Chapter 4 baselines.

The point of this module is comparability: MitM, LToS and P-MADDPG must run
inside the *same* environment object as QOLU + BUIA, under the same scale
settings (``grid_size``, ``conversion_fraction``,
``max_conversions_per_iteration``), from the same seeds, and be scored with the
same metrics from :mod:`chapter4_qolu.env.evaluation`.  A method therefore never
builds or configures an environment; it is handed one in :meth:`reset` and may
only read it.

:func:`run_baseline` mirrors :func:`chapter4_qolu.runner.train_joint`'s ordering
and reporting so the two entry points produce directly comparable records:

1. the environment draws its attractiveness-weighted decision buffer (Eq. 2);
2. the method appraises every candidate and optionally narrows the set of
   parcels it puts forward (:class:`BaselineDecision`);
3. the approved proposals are applied in one synchronised update;
4. the method observes the outcome and takes one learning step.

``train_joint`` is deliberately separate: baselines are a *parallel* entry
point, so no baseline can change QOLU + BUIA behaviour.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from chapter4_qolu.env.evaluation import evaluate_landscape
from chapter4_qolu.env.parcel_grid import ParcelIndex
from chapter4_qolu.env.urban_env import (
    APPROVE,
    REJECT,
    DecisionBatch,
    StepResult,
    UrbanLandUseEnv,
)
from chapter4_qolu.land_cover import LandCoverGroup

__all__ = [
    "BaselineDecision",
    "BaselineMethod",
    "BaselineReport",
    "conversion_budget",
    "run_baseline",
    "top_k_by_score",
]


def conversion_budget(env: UrbanLandUseEnv) -> int | None:
    """The shared per-iteration conversion budget ``B``.

    ``None`` means the environment applies no cap, i.e. every approved
    candidate converts.  Methods that rank their own decisions (LToS, MitM)
    must honour this number so that all methods develop land at the same rate.
    """
    return env.config.max_conversions_per_iteration


@dataclass(frozen=True)
class BaselineDecision:
    """What a method returns for one decision buffer.

    Args:
        decisions: one appraisal per candidate *in buffer order*, ``REJECT``
            or ``APPROVE`` (the published QOLU action set, reused so the
            environment step is byte-for-byte the same for every method).
        proposals: the parcels the method's bottom-up layer put forward, if it
            has one.  ``None`` - the default - means "every candidate was
            proposed", which is what the environment does for a method without
            an investor.
    """

    decisions: np.ndarray
    proposals: tuple[ParcelIndex, ...] | None = None

    def __post_init__(self) -> None:
        actions = np.asarray(self.decisions, dtype=np.int64).reshape(-1)
        if np.ndim(self.decisions) > 1:
            raise ValueError("decisions must be a one-dimensional array")
        object.__setattr__(self, "decisions", actions)
        if self.proposals is not None:
            object.__setattr__(self, "proposals", tuple(self.proposals))


class BaselineMethod(ABC):
    """A pluggable decision-making method for the land-use environment.

    Contract:

    * :meth:`reset` binds the method to an environment and clears *episode*
      state.  Learned parameters must survive a reset, because the driver calls
      it again whenever the environment terminates mid-run - exactly as
      ``train_joint`` keeps its agents across episode boundaries.
    * :meth:`decide` sees one decision buffer and returns a
      :class:`BaselineDecision`.
    * :meth:`observe` and :meth:`train_step` are optional hooks so that
      non-learning methods stay short.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in reports and result files."""

    @abstractmethod
    def reset(self, env: UrbanLandUseEnv, seed: int | None = None) -> None:
        """Bind to ``env`` for a new episode (parameters are kept)."""

    @abstractmethod
    def decide(self, batch: DecisionBatch) -> BaselineDecision:
        """Appraise every candidate parcel of ``batch``."""

    def observe(self, result: StepResult) -> None:
        """Record the outcome of the synchronised update (default: no-op)."""

    def train_step(self) -> float | None:
        """One learning update; returns a loss, or ``None`` when there is none."""
        return None


@dataclass(frozen=True)
class BaselineReport:
    """Per-iteration diagnostics of a baseline run.

    Mirrors :class:`chapter4_qolu.runner.JointTrainingReport` field for field
    (minus the QOLU/BUIA-specific series) so the two can be tabulated together.
    """

    method: str
    iterations: int
    episodes: int
    team_rewards: tuple[float, ...]
    losses: tuple[float, ...]
    urban_fraction: tuple[float, ...]
    conversion_alignment: tuple[float, ...]
    well_being: tuple[float, ...]
    n_conversions: tuple[int, ...]
    n_proposals: tuple[int, ...]
    n_approved: tuple[int, ...]
    n_candidates: tuple[int, ...]
    final_metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def mean_team_reward(self) -> float:
        return float(np.mean(self.team_rewards)) if self.team_rewards else 0.0

    @property
    def total_conversions(self) -> int:
        return int(sum(self.n_conversions))


def _validated_actions(decision: BaselineDecision, batch: DecisionBatch) -> np.ndarray:
    """Check a method's output before it reaches the environment."""
    actions = np.asarray(decision.decisions, dtype=np.int64).reshape(-1)
    if actions.size != len(batch.parcels):
        raise ValueError(
            f"method returned {actions.size} decisions for "
            f"{len(batch.parcels)} candidate parcels"
        )
    if not np.isin(actions, (REJECT, APPROVE)).all():
        raise ValueError(
            f"decisions must be REJECT ({REJECT}) or APPROVE ({APPROVE}), "
            f"got {sorted(set(actions.tolist()))}"
        )
    return actions


def _proposals(
    decision: BaselineDecision,
) -> Iterable[ParcelIndex] | None:
    return None if decision.proposals is None else tuple(decision.proposals)


def run_baseline(
    env: UrbanLandUseEnv,
    method: BaselineMethod,
    iterations: int,
    seed: int | None = None,
) -> BaselineReport:
    """Run ``method`` for ``iterations`` synchronised updates of ``env``.

    The environment is reset with ``seed`` exactly as ``train_joint`` does, so a
    baseline and the published system see the same decision buffers for the same
    seed until their decisions start to differ.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")

    batch = env.reset(seed=seed)
    method.reset(env, seed)
    episodes = 1

    team_rewards: list[float] = []
    losses: list[float] = []
    urban_fraction: list[float] = []
    alignment: list[float] = []
    well_being: list[float] = []
    n_conversions: list[int] = []
    n_proposals: list[int] = []
    n_approved: list[int] = []
    n_candidates: list[int] = []

    for iteration in range(iterations):
        decision = method.decide(batch)
        actions = _validated_actions(decision, batch)
        result = env.step(actions, proposals=_proposals(decision))

        method.observe(result)
        loss = method.train_step()
        if loss is not None:
            losses.append(float(loss))

        team_rewards.append(float(result.team_reward))
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
            # evaluation metrics below have to measure (same rule as train_joint).
            episodes += 1
            episode_seed = None if seed is None else seed + episodes
            batch = env.reset(seed=episode_seed)
            method.reset(env, episode_seed)
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

    return BaselineReport(
        method=method.name,
        iterations=iterations,
        episodes=episodes,
        team_rewards=tuple(team_rewards),
        losses=tuple(losses),
        urban_fraction=tuple(urban_fraction),
        conversion_alignment=tuple(alignment),
        well_being=tuple(well_being),
        n_conversions=tuple(n_conversions),
        n_proposals=tuple(n_proposals),
        n_approved=tuple(n_approved),
        n_candidates=tuple(n_candidates),
        final_metrics=final_metrics,
    )


def top_k_by_score(
    parcels: Sequence[ParcelIndex], scores: np.ndarray, budget: int | None
) -> tuple[ParcelIndex, ...]:
    """The ``budget`` best *distinct* parcels, highest score first.

    Shared by the two methods that have to enforce the global conversion budget
    themselves (LToS ranks by Q-value, MitM by compromise-programming score).
    The decision buffer is sampled *with replacement*, so a parcel appearing
    several times keeps its best score and is counted once.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size != len(parcels):
        raise ValueError(f"got {values.size} scores for {len(parcels)} parcels")
    best: dict[ParcelIndex, float] = {}
    for parcel, score in zip(parcels, values):
        current = best.get(parcel)
        if current is None or score > current:
            best[parcel] = float(score)
    ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))
    if budget is not None:
        ranked = ranked[: max(int(budget), 0)]
    return tuple(parcel for parcel, _ in ranked)
