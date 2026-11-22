"""MitM baseline - "modeling-in-the-middle" (Bone, Dragicevic & White, 2011).

**Source: the paper is paywalled; this implementation follows its abstract
and reference list.**  The abstract describes agents representing "households
and commercial enterprises" that "select locations to inhabit based on
population densities and attractivity preferences", whose land-use change "is
evaluated by a set of agents representing different stakeholders who are
embedded with RL algorithms that allow them to influence the land use change
process so that their objectives are addressed".  Every mechanism below is
therefore inferred - marked INFERRED in the research note - and each
``Design note`` is recorded here or in
:mod:`chapter4_qolu.baselines.mitm_allocation`.

The mapping is unusually direct because the published Chapter 4 environment was
explicitly designed after MitM ("inspired by the work of Bone et al. [2011]"):

======================================  ====================================
MitM element                            Environment element
======================================  ====================================
attractivity surface                    ``env/attractiveness.py`` (Eq. 1),
                                        wrapped in the White-Engelen form
households picking by attractivity      the *myopic greedy* chooser here -
                                        **not** the neural BUIA, which would
                                        turn MitM into QOLU + BUIA
stakeholder agents with RL              one tabular TD(0) learner per
                                        ``TopDownObjective``
stakeholder reward ``O_m``              the published objective reward terms
MODM aggregation                        compromise programming over the
                                        scenario weights
conversion budget                       the shared
                                        ``max_conversions_per_iteration``
======================================  ====================================

The learner is **tabular / linear only**: one scalar value ``V_m`` and one
3-vector of criterion weights ``theta_m`` per stakeholder.  Nothing here is a
neural network, and nothing here imports torch - MitM is a 2011 GIS/ABM method
and upgrading it would silently strengthen the baseline.

Design note (which ``theta`` variant is the headline): the note offers two.  The
default freezes ``theta_m`` as a one-hot on stakeholder ``m``'s own objective,
so the aggregate MCE weighting is *exactly* the scenario weights and the
four-scenario axis keeps its meaning.  ``learn_weights=True`` (the
:func:`learned_mitm` variant) lets each stakeholder nudge its own ``theta_m``,
which is truer to MitM - it negotiates weights endogenously - but lets the
aggregate drift away from the scenario being reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from chapter4_qolu.baselines.base import (
    BaselineDecision,
    BaselineMethod,
    conversion_budget,
)
from chapter4_qolu.baselines.config import MitMConfig
from chapter4_qolu.baselines.mitm_allocation import (
    compromise_allocation,
    criterion_matrix,
    density_discount,
    greedy_choice,
    white_engelen_attractivity,
)
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.env.urban_env import (
    APPROVE,
    REJECT,
    DecisionBatch,
    StepResult,
    UrbanLandUseEnv,
)
from chapter4_qolu.objectives import TopDownObjective

__all__ = ["MitMMethod", "MitMStakeholder", "learned_mitm"]

_N_CRITERIA = len(TopDownObjective)


@dataclass
class MitMStakeholder:
    """One stakeholder: a TD(0) value and a criterion-weight vector ``theta_m``.

    Design note (learning rule): the paper cites Sutton (1988) and Barto & Sutton
    (1981) but **no Watkins / Q-learning**, so the learner is TD(0) value
    estimation plus an associative-search nudge over the criterion weights,
    not Q-learning.  The nudge moves ``theta_m`` toward the criterion profile
    that was actually realised when the round beat the running value, and away
    from it when the round disappointed:

        V_m     <- V_m + alpha * (R_m - V_m)
        theta_m <- proj( theta_m + eta * (R_m - V_m) * (x_m - theta_m) )

    where ``x_m`` is the mean criterion profile of the converted parcels and
    ``proj`` clips to non-negative values and renormalises onto the simplex.
    """

    objective: TopDownObjective
    criterion_weights: np.ndarray
    value: float = 0.0

    def __post_init__(self) -> None:
        weights = np.asarray(self.criterion_weights, dtype=np.float64).reshape(-1)
        if weights.size != _N_CRITERIA:
            raise ValueError(f"criterion weights must have {_N_CRITERIA} entries")
        self.criterion_weights = _project(weights)
        self.value = float(self.value)

    def observe(self, reward: float, rate: float) -> float:
        """One TD(0) step; returns the TD error ``R_m - V_m``."""
        error = float(reward) - self.value
        self.value += float(rate) * error
        return error

    def nudge(self, error: float, realised: np.ndarray, rate: float) -> None:
        """Move ``theta_m`` toward (or away from) the realised criterion profile."""
        profile = np.asarray(realised, dtype=np.float64).reshape(-1)
        if profile.size != _N_CRITERIA:
            raise ValueError(f"realised profile must have {_N_CRITERIA} entries")
        self.criterion_weights = _project(
            self.criterion_weights
            + float(rate) * float(error) * (profile - self.criterion_weights)
        )


def _project(weights: np.ndarray) -> np.ndarray:
    """Clip to non-negative values and renormalise onto the simplex."""
    clipped = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(_N_CRITERIA, 1.0 / _N_CRITERIA)
    return clipped / total


@dataclass
class _Round:
    """What :meth:`MitMMethod.observe` hands to :meth:`MitMMethod.train_step`."""

    rewards: dict[TopDownObjective, float] = field(default_factory=dict)
    realised: np.ndarray = field(
        default_factory=lambda: np.full(_N_CRITERIA, 1.0 / _N_CRITERIA)
    )


class MitMMethod(BaselineMethod):
    """The MitM loop as a pluggable :class:`BaselineMethod`."""

    def __init__(
        self, config: MitMConfig | None = None, seed: int | None = None
    ) -> None:
        self.config = config or MitMConfig()
        self.rng = np.random.default_rng(seed)
        self.stakeholders = [
            MitMStakeholder(
                objective=objective,
                criterion_weights=np.eye(_N_CRITERIA)[index],
            )
            for index, objective in enumerate(TopDownObjective)
        ]
        self._env: UrbanLandUseEnv | None = None
        self._grid_before: ParcelGrid | None = None
        self._pending: _Round | None = None

    @classmethod
    def for_env(
        cls, env: UrbanLandUseEnv, seed: int | None = None, **overrides: object
    ) -> MitMMethod:
        """Build a method for ``env`` (MitM needs no shapes, only parameters)."""
        return cls(MitMConfig(**overrides), seed=seed)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        return "mitm_learned" if self.config.learn_weights else "mitm"

    @property
    def criterion_weights(self) -> np.ndarray:
        """The aggregate MCE weighting ``sum_m scenario_m * theta_m``.

        With the frozen (headline) variant every ``theta_m`` is a one-hot on
        stakeholder ``m``'s own objective, so this is exactly the normalised
        scenario weighting of Table 3.
        """
        if self._env is None:
            raise RuntimeError("call reset() before reading the criterion weights")
        scenario = self._env.weights
        total = np.zeros(_N_CRITERIA, dtype=np.float64)
        for index, objective in enumerate(TopDownObjective):
            total += (
                float(scenario[objective]) * self.stakeholders[index].criterion_weights
            )
        return _project(total)

    # ------------------------------------------------------------------
    # the MitM round: attractivity -> bottom-up choice -> MODM allocation
    # ------------------------------------------------------------------
    def reset(self, env: UrbanLandUseEnv, seed: int | None = None) -> None:
        self._env = env
        self._grid_before = None
        self._pending = None
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def decide(self, batch: DecisionBatch) -> BaselineDecision:
        if self._env is None:
            raise RuntimeError("call reset() before decide()")
        env = self._env
        self._grid_before = env.grid

        candidates = tuple(dict.fromkeys(batch.parcels))
        criteria = criterion_matrix(env.grid, candidates, env.config.reward)
        weights = self.criterion_weights

        # 1. attractivity surface, White-Engelen form
        neighbourhood = np.asarray(
            [env.attractiveness.at(parcel) for parcel in candidates], dtype=np.float64
        )
        perturbation = self.rng.uniform(
            -self.config.perturbation_scale,
            self.config.perturbation_scale,
            size=len(candidates),
        )
        attractivity = white_engelen_attractivity(
            suitability=criteria @ weights,
            neighbourhood=neighbourhood,
            neighbourhood_weight=self.config.neighbourhood_weight,
            perturbation=perturbation,
        )

        # 2. myopic bottom-up chooser (households / enterprises)
        budget = conversion_budget(env)
        wanted = (
            len(candidates)
            if budget is None
            else min(len(candidates), self.config.proposal_multiplier * budget)
        )
        proposals = greedy_choice(
            attractivity
            * density_discount(env.grid, candidates, self.config.density_pressure),
            candidates,
            wanted,
        )

        # 3. the "middle": MODM allocation of the top-B proposals
        positions = [candidates.index(parcel) for parcel in proposals]
        allocated = set(
            compromise_allocation(
                proposals,
                criteria[positions],
                weights,
                budget=budget,
                order=self.config.compromise_order,
            )
        )
        decisions = np.asarray(
            [APPROVE if parcel in allocated else REJECT for parcel in batch.parcels],
            dtype=np.int64,
        )
        return BaselineDecision(decisions=decisions, proposals=proposals)

    # ------------------------------------------------------------------
    # stakeholder learning
    # ------------------------------------------------------------------
    def observe(self, result: StepResult) -> None:
        if self._grid_before is None or self._env is None:
            raise RuntimeError("call decide() before observe()")
        converted = tuple(conversion.parcel for conversion in result.conversions)
        self._pending = _Round(
            rewards=dict(result.objective_rewards),
            realised=self._realised_profile(converted),
        )

    def _realised_profile(self, converted: tuple[ParcelIndex, ...]) -> np.ndarray:
        """``x_m``: the criterion mix the realised pattern actually delivered."""
        assert self._grid_before is not None and self._env is not None
        if not converted:
            # Nothing was built, so there is no realised profile to move toward.
            return np.full(_N_CRITERIA, 1.0 / _N_CRITERIA)
        criteria = criterion_matrix(
            self._grid_before, converted, self._env.config.reward
        )
        return _project(criteria.mean(axis=0))

    def train_step(self) -> float | None:
        """TD(0) on every stakeholder; returns the mean absolute TD error."""
        if self._pending is None:
            return None
        errors = []
        for stakeholder in self.stakeholders:
            error = stakeholder.observe(
                self._pending.rewards[stakeholder.objective],
                self.config.value_learning_rate,
            )
            if self.config.learn_weights:
                stakeholder.nudge(
                    error, self._pending.realised, self.config.weight_learning_rate
                )
            errors.append(abs(error))
        self._pending = None
        return float(np.mean(errors))


def learned_mitm(
    env: UrbanLandUseEnv, seed: int | None = None, **overrides: object
) -> MitMMethod:
    """The secondary variant: stakeholders negotiate their criterion weights."""
    config = replace(MitMConfig(**overrides), learn_weights=True)  # type: ignore[arg-type]
    method = MitMMethod(config, seed=seed)
    method.reset(env, seed)
    return method
