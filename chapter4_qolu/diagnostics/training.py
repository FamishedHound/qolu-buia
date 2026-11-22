"""What the run being diagnosed actually *did*, beside what it predicted.

A diagnostics run has two halves.  The rollout measures the stakeholders'
predictions - the approval rate, the CRPS, the PIT.  The training that produced
those stakeholders returns a
:class:`~chapter4_qolu.runner.JointTrainingReport` full of the numbers the
chapter reports - the conversion count and the three published metrics.

Keeping only the first half would mean the approval rate and the metrics of
"the same run" come from two files written by two invocations, and only the
agreement of their *settings* would say they are the same run.  This module
records both halves in one document.

It also records how many times ``theta^- <- theta`` actually fired.
``target_sync_interval`` is a free choice of the implementation, and a value
longer than a run's whole gradient budget disables the double-Q target
entirely without changing a single line of the config.  ``learner_record``
makes that visible as a count rather than as an arithmetic exercise on two
other numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chapter4_qolu.run_record import absolute_metrics, learner_record

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from chapter4_qolu.env.urban_env import UrbanLandUseEnv
    from chapter4_qolu.runner import JointTrainingReport
    from chapter4_qolu.scenario_worker import RunSpec

__all__ = ["learner_record", "training_record"]


def training_record(
    report: JointTrainingReport | None,
    env: UrbanLandUseEnv | None,
    spec: RunSpec | None,
) -> dict[str, Any]:
    """The outcome of the run that produced the agents, or a note that it was read.

    ``{"source": "checkpoint"}`` is the honest answer when the stakeholders
    came off disk: this process trained nothing, so it has no conversion count
    and no final landscape to measure, and inventing one from the evaluation
    rollout would report a different experiment under the same key.
    """
    if report is None or env is None or spec is None:
        return {"source": "checkpoint"}
    return {
        "source": "trained",
        "iterations": int(report.iterations),
        "episodes": int(report.episodes),
        "total_conversions": int(sum(report.n_conversions)),
        "mean_team_reward": float(report.mean_team_reward),
        "final_metrics": dict(report.final_metrics),
        "final_metrics_absolute": absolute_metrics(env, spec.max_distance_km),
        "absolute_max_distance_km": float(spec.max_distance_km),
    }
