"""What one evaluation rollout logs for every appraised candidate.

One row per appraisal, and nothing summarised at collection time: the predicted
categorical probabilities for the action that was actually taken, the atom
support they sit on, the predicted mean, the quality score the decision was
taken on, the reward the stakeholder was paid, the agent's own bootstrap value
for the state the transition landed in, and the same prediction from an
untrained copy of the network.  Every statistic in this package is a function of
those columns, so a disputed number can be recomputed from the log without
re-running anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chapter4_qolu.diagnostics.returns import (
    TARGET_DESCRIPTIONS,
    n_step_reward_sums,
    n_step_targets,
    tail_returns,
)

__all__ = [
    "HEADLINE_TARGET",
    "NO_SPREAD_RATIO",
    "AppraisalLog",
    "DiagnosticsLog",
    "quality_score_record",
]

#: Design note: the bootstrap-free tail is the headline because it is the only
#: one of the three readings that contains no output of the model being scored.
#: The other two are reported beside it, never instead of it.
HEADLINE_TARGET: str = "tail_return"

#: A quality-score spread this small *relative to the typical magnitude of the
#: score itself* is called no spread at all: the policy puts every candidate it
#: was shown at the same place, so its approval rate is a property of the
#: network's bias and not of the states it appraised.
#:
#: Design note (why a ratio and not an absolute): ``Q`` lives on the atom
#: support, which Table 1 sets to [-200, 200], and a run whose scores
#: sit around 100 is a different scale from one whose scores sit around 0.001.
#: The threshold is a reporting convenience; the raw ``sd``, ``min`` and ``max``
#: are always reported beside the boolean so a reader can set their own.
NO_SPREAD_RATIO: float = 1e-3


def _within_iteration_sd(values: np.ndarray, iteration_index: np.ndarray) -> float:
    """Mean over iterations of the spread *inside* one decision buffer.

    The landscape moves between iterations, so a policy can vary across a
    rollout while ranking every candidate of a single decision buffer
    identically.  Only the second kind of variation can change which proposal
    the consensus approves, so it is measured separately.  Iterations holding a
    single appraisal have no spread to measure and are skipped; ``nan`` when
    none holds two.
    """
    spreads = [
        float(values[iteration_index == index].std())
        for index in np.unique(iteration_index)
        if int(np.count_nonzero(iteration_index == index)) > 1
    ]
    return float(np.mean(spreads)) if spreads else float("nan")


def quality_score_record(
    quality_scores: np.ndarray, iteration_index: np.ndarray
) -> dict[str, Any]:
    """How much the quality score varied across the candidates appraised.

    ``Q(s, approve) - Q(s, reject)`` is the number the consensus sums and whose
    sign is the greedy action, so its *spread* is the difference between a
    policy that discriminates between candidates and one that has collapsed onto
    a constant.  An approval rate cannot tell those apart: a stakeholder that
    approves everything because it separates nothing and one that approves
    everything on the merits report the same 1.0.
    """
    values = np.asarray(quality_scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("a quality-score record needs at least one appraisal")
    sd = float(values.std())
    mean_absolute = float(np.abs(values).mean())
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": sd,
        "min": float(values.min()),
        "max": float(values.max()),
        "mean_absolute": mean_absolute,
        "within_iteration_sd": _within_iteration_sd(values, iteration_index),
        "spread_ratio": (sd / mean_absolute if mean_absolute > 0.0 else float("nan")),
        "fraction_positive": float(np.mean(values > 0.0)),
        "no_spread": bool(sd <= NO_SPREAD_RATIO * mean_absolute),
    }


@dataclass(frozen=True)
class AppraisalLog:
    """Every appraisal one stakeholder made during an evaluation rollout."""

    stakeholder: int
    objective: str
    atoms: np.ndarray
    #: ``(n, n_atoms)`` - ``p(s, a)`` for the action actually taken.
    probabilities: np.ndarray
    #: The same, from a freshly initialised copy of the network.
    reference_probabilities: np.ndarray
    actions: np.ndarray
    #: ``Q(s, approve) - Q(s, reject)`` - the paper's quality metric, the number
    #: the consensus sums and whose sign is the greedy action.
    quality_scores: np.ndarray
    rewards: np.ndarray
    #: ``Q(s', a*)`` under double Q-learning, for the n-step training target.
    bootstrap_values: np.ndarray
    iteration_index: np.ndarray
    gamma: float
    n_step: int

    def __post_init__(self) -> None:
        rows = self.probabilities.shape
        if len(rows) != 2:
            raise ValueError("probabilities must have shape (n, n_atoms)")
        if rows[1] != self.atoms.size:
            raise ValueError("probabilities and the atom support disagree")
        for name in (
            "reference_probabilities",
            "actions",
            "quality_scores",
            "rewards",
            "bootstrap_values",
            "iteration_index",
        ):
            if getattr(self, name).shape[0] != rows[0]:
                raise ValueError(f"{name} must have one entry per appraisal")

    @property
    def size(self) -> int:
        return int(self.probabilities.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.atoms.size)

    @property
    def predicted_mean(self) -> np.ndarray:
        """``E[Z(s, a)]`` - the only part of the forecast the policy reads."""
        return self.probabilities @ self.atoms

    def realised_returns(self) -> dict[str, np.ndarray]:
        """The three readings of "what actually happened"; see ``returns.py``."""
        return {
            "n_step_reward_sum": n_step_reward_sums(
                self.rewards, self.gamma, self.n_step
            ),
            "n_step_target": n_step_targets(
                self.rewards, self.gamma, self.n_step, self.bootstrap_values
            ),
            "tail_return": tail_returns(self.rewards, self.gamma),
        }

    def as_record(self) -> dict[str, Any]:
        """Scalar description of the log, for the summary JSON."""
        return {
            "stakeholder": int(self.stakeholder),
            "objective": self.objective,
            "n_appraisals": self.size,
            "n_atoms": self.n_atoms,
            "v_min": float(self.atoms[0]),
            "v_max": float(self.atoms[-1]),
            "gamma": float(self.gamma),
            "n_step": int(self.n_step),
            "approval_rate": float(np.mean(self.actions == 1)),
            "mean_reward": float(self.rewards.mean()),
            # The approval rate says how often the stakeholder said yes; this
            # says whether it was answering a question about the candidate.
            "quality_score": quality_score_record(
                self.quality_scores, self.iteration_index
            ),
        }


@dataclass(frozen=True)
class DiagnosticsLog:
    """Every stakeholder's appraisal log, plus what the rollout was."""

    logs: tuple[AppraisalLog, ...]
    iterations: int
    samples_per_iteration: int
    qolu_reward_mode: str
    scenario_weights: tuple[float, ...]
    epsilon: float = 0.0
    investor: bool = True
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.logs:
            raise ValueError("at least one stakeholder log is required")

    @property
    def atoms(self) -> np.ndarray:
        return self.logs[0].atoms

    def pooled(self) -> AppraisalLog:
        """Every stakeholder's appraisals concatenated into one log.

        Design note: pooling is legitimate for the scoring statistics, which
        only ever see (forecast, observation) pairs, and it is what produces a
        single headline number for the chapter.  The n-step and tail returns are
        computed *before* pooling, per stakeholder, because each stakeholder has
        its own reward stream; the pooled log's own ``rewards`` are therefore
        only meaningful stakeholder by stakeholder, which is why every caller
        passes the realised returns in explicitly.
        """
        return AppraisalLog(
            stakeholder=-1,
            objective="pooled",
            atoms=self.atoms,
            probabilities=np.concatenate([log.probabilities for log in self.logs]),
            reference_probabilities=np.concatenate(
                [log.reference_probabilities for log in self.logs]
            ),
            actions=np.concatenate([log.actions for log in self.logs]),
            quality_scores=np.concatenate([log.quality_scores for log in self.logs]),
            rewards=np.concatenate([log.rewards for log in self.logs]),
            bootstrap_values=np.concatenate(
                [log.bootstrap_values for log in self.logs]
            ),
            iteration_index=np.concatenate([log.iteration_index for log in self.logs]),
            gamma=self.logs[0].gamma,
            n_step=self.logs[0].n_step,
        )

    def pooled_returns(self) -> dict[str, np.ndarray]:
        """The three targets, computed per stakeholder and then concatenated."""
        per_stakeholder = [log.realised_returns() for log in self.logs]
        return {
            name: np.concatenate([entry[name] for entry in per_stakeholder])
            for name in TARGET_DESCRIPTIONS
        }

    def as_record(self) -> dict[str, Any]:
        return {
            "iterations": int(self.iterations),
            "samples_per_iteration": int(self.samples_per_iteration),
            "n_stakeholders": len(self.logs),
            "qolu_reward_mode": self.qolu_reward_mode,
            "scenario_weights": [float(value) for value in self.scenario_weights],
            "epsilon": float(self.epsilon),
            "investor": bool(self.investor),
            "notes": list(self.notes),
            "stakeholders": [log.as_record() for log in self.logs],
        }
