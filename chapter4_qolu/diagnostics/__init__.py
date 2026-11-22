"""Diagnostics for the QOLU categorical return distribution.

Two questions are asked of a distributional agent: is the distribution used in
the decisions, and - since the accuracy and error of the learnt ``Q``
distributions can be measured - is an internal model of the return learnt?

This package answers both.

**Is the distribution influencing the decision?**  No, and the code says so:
every decision reads ``Z(s, a)`` only through ``E[Z]`` - see the Design note at
the top of :mod:`chapter4_qolu.agent.qolu_agent` and the statement in
:data:`chapter4_qolu.diagnostics.summary.POLICY_USE_OF_THE_DISTRIBUTION`, which
is pinned by a behavioural test rather than asserted in prose.

**Is an internal model nevertheless learnt?**  That is a measurement, and it is
what the rest of this package does: given a trained agent (loaded from a
checkpoint written by :mod:`chapter4_qolu.agent.checkpoint`) and an evaluation
rollout, it logs the predicted categorical probabilities for every appraised
candidate and scores them with

* the MAE and RMSE of ``E[Z]``, beside the quantity that drives decisions;
* the **randomised** probability integral transform and its Kolmogorov-Smirnov
  statistic against ``U(0, 1)``;
* the empirical coverage of the central 50 % and 90 % intervals;
* the CRPS, and its **skill score against two references** - the network at
  initialisation and a state-blind climatology.  Positive skill against
  climatology is the evidence that ``Z`` is state-conditional, and is the single
  headline number;
* the boundary-atom mass, the standard C51 support pathology.

A negative result is a complete answer and is reported in the same words as a
positive one.
"""

from __future__ import annotations

from chapter4_qolu.diagnostics.calibration import (
    COVERAGE_LEVELS,
    interval_coverage,
    kolmogorov_smirnov_uniform,
    randomised_pit,
)
from chapter4_qolu.diagnostics.collect import collect_appraisals
from chapter4_qolu.diagnostics.crps import (
    climatology_probabilities,
    crps,
    crps_skill_score,
)
from chapter4_qolu.diagnostics.errors import (
    boundary_mass,
    error_statistics,
    target_clustering,
)
from chapter4_qolu.diagnostics.figure import calibration_figure
from chapter4_qolu.diagnostics.records import (
    HEADLINE_TARGET,
    AppraisalLog,
    DiagnosticsLog,
)
from chapter4_qolu.diagnostics.report import render_diagnostics_report
from chapter4_qolu.diagnostics.returns import (
    TARGET_DESCRIPTIONS,
    n_step_reward_sums,
    n_step_targets,
    tail_returns,
)
from chapter4_qolu.diagnostics.summary import (
    CALIBRATION_CAVEAT,
    POLICY_USE_OF_THE_DISTRIBUTION,
    diagnostics_summary,
)
from chapter4_qolu.diagnostics.sweep import run_sweep, sweep_cells, sweep_paths
from chapter4_qolu.diagnostics.sweep_report import render_sweep_report
from chapter4_qolu.diagnostics.sweep_summary import aggregate_records, cell_record

__all__ = [
    "CALIBRATION_CAVEAT",
    "COVERAGE_LEVELS",
    "HEADLINE_TARGET",
    "POLICY_USE_OF_THE_DISTRIBUTION",
    "TARGET_DESCRIPTIONS",
    "AppraisalLog",
    "DiagnosticsLog",
    "aggregate_records",
    "boundary_mass",
    "calibration_figure",
    "cell_record",
    "climatology_probabilities",
    "collect_appraisals",
    "crps",
    "crps_skill_score",
    "diagnostics_summary",
    "error_statistics",
    "interval_coverage",
    "kolmogorov_smirnov_uniform",
    "n_step_reward_sums",
    "n_step_targets",
    "randomised_pit",
    "render_diagnostics_report",
    "render_sweep_report",
    "run_sweep",
    "sweep_cells",
    "sweep_paths",
    "tail_returns",
    "target_clustering",
]
