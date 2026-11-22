"""Measure the accuracy and the error of the QOLU return distributions.

This driver states how distributional RL is - and is not - used in the final
algorithm, and *measures* the learnt ``Q`` distributions, to show whether an
internal model is learnt.  It produces both halves as one document: the
statement that the decision
path reads ``Z(s, a)`` only through ``E[Z]``, and the statistics that say
whether ``Z`` itself is any good.

Usage::

    # print the plan and the ETA, run nothing
    python -m chapter4_qolu.run_distribution_diagnostics --raster <raster> --dry-run

    # train a run, persist its agents, diagnose them
    python -m chapter4_qolu.run_distribution_diagnostics --raster <raster> \\
        --window E495_N140 --scenario 1 --seed 0 --iterations 1000 --resume

    # diagnose agents a campaign already persisted
    python -m chapter4_qolu.run_distribution_diagnostics --raster <raster> \\
        --campaign results/campaign_main --scenario 1 --seed 0

    # diagnose one directory of checkpoints directly
    python -m chapter4_qolu.run_distribution_diagnostics --raster <raster> \\
        --checkpoint results/campaign_main/50x50_f1.0_cap10/scenario1/\\
qolu_buia_seed0.agents

Output lands in ``results/distribution_diagnostics_<tag>/``:
``distribution_diagnostics.json`` (every statistic, full precision),
``distribution_diagnostics.md`` (the tables and the verdict), ``agents/`` (the
checkpoints the diagnosis was run on) and ``figures/`` (the two-panel PIT and
reliability figure, with its numbers as CSV beside it).

Design note (the region mask is not exposed): the evaluation environment is the
unclipped window box.  A boundary clip only ever removes land from the window,
and the diagnostic is a property of the agent's predictions on the states it is
shown; the window that was used is recorded in the summary either way.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chapter4_qolu.agent.checkpoint import agent_checkpoint_paths, load_agents
from chapter4_qolu.agent.qolu_agent import QOLUAgent
from chapter4_qolu.diagnostics.arguments import build_parser, is_sweep, settings_from
from chapter4_qolu.diagnostics.collect import collect_appraisals
from chapter4_qolu.diagnostics.figure import calibration_figure
from chapter4_qolu.diagnostics.plan import (
    DiagnosticsPaths,
    DiagnosticsSettings,
    agent_source,
    build_environment,
    diagnostics_paths,
    investor_for,
    is_complete,
    run_plan,
)
from chapter4_qolu.diagnostics.report import render_diagnostics_report
from chapter4_qolu.diagnostics.summary import diagnostics_summary
from chapter4_qolu.diagnostics.sweep import (
    DEFAULT_SWEEP_SCENARIOS,
    DEFAULT_SWEEP_SEEDS,
    run_sweep,
    sweep_paths,
)
from chapter4_qolu.diagnostics.training import learner_record, training_record
from chapter4_qolu.runner import train_joint
from chapter4_qolu.scenario_worker import RunSpec, qolu_config_for

__all__ = [
    "DiagnosticsPaths",
    "DiagnosticsSettings",
    "diagnostics_paths",
    "main",
    "run_distribution_diagnostics",
    "run_sweep",
    "sweep_paths",
]


def _build_parser() -> argparse.ArgumentParser:
    """Every flag this driver takes; see :mod:`diagnostics.arguments`."""
    return build_parser(__doc__)


logger = logging.getLogger(__name__)


#: Design note (the investor is never the trained one).  BUIA has no
#: checkpoint, so a run diagnosed from disk can only be evaluated against a
#: freshly seeded investor.  Rather than have the two paths differ - a trained
#: proposer when the run happened to train in-process, an untrained one when it
#: loaded - the evaluation rollout *always* builds a fresh investor, so the two
#: are comparable and the summary can say so in one word.  What that changes is
#: *which* parcels are appraised, not the property being measured: the
#: calibration of a stakeholder's prediction on the states it is actually
#: shown.  It is a real limitation and it is recorded, not hidden: the rollout's
#: candidate parcels are drawn under an untrained proposal policy.
INVESTOR_STATE: str = "fresh"


def _train_agents(
    settings: DiagnosticsSettings, spec: RunSpec, paths: DiagnosticsPaths
) -> tuple[list[QOLUAgent], dict[str, Any]]:
    """Run the published joint loop once, persist the stakeholders, keep the report.

    ``settings.qolu_overrides`` reaches the learner through
    :func:`~chapter4_qolu.scenario_worker.qolu_config_for`, the same resolver
    :mod:`chapter4_qolu.hparam_worker` perturbs through, so a diagnosed run and
    a sensitivity run of the same settings are the same experiment.
    """
    env = build_environment(spec)
    config = qolu_config_for(
        spec.method,
        observation_channels=env.observation_channels,
        window=env.config.observation_window,
        overrides=settings.qolu_overrides,
    )
    stakeholders = [
        QOLUAgent(config, seed=spec.seed + index) for index in range(spec.stakeholders)
    ]
    investor = investor_for(env, spec)
    report = train_joint(
        env,
        stakeholders,
        investor if settings.investor else None,
        iterations=spec.iterations,
        seed=spec.seed,
        qolu_reward=spec.qolu_reward_config(),
        checkpoint_dir=paths.agents,
    )
    return stakeholders, training_record(report, env, spec)


def _resolve_agents(
    settings: DiagnosticsSettings, spec: RunSpec, paths: DiagnosticsPaths
) -> tuple[list[QOLUAgent], Path, str, dict[str, Any]]:
    """The stakeholders to diagnose, and an honest label for where they came from."""
    source = agent_source(settings, paths)
    if source is None:
        stakeholders, training = _train_agents(settings, spec, paths)
        return stakeholders, paths.agents, "trained", training
    return (
        list(load_agents(source)),
        Path(source),
        "checkpoint",
        training_record(None, None, None),
    )


def run_distribution_diagnostics(
    settings: DiagnosticsSettings, dry_run: bool = False
) -> dict[str, Any]:
    """Train or load, roll out, score, and write the JSON, figure and markdown."""
    torch.set_num_threads(max(1, int(settings.torch_threads)))
    paths = diagnostics_paths(settings)
    if settings.resume and is_complete(paths.summary):
        # The JSON is the record and the markdown is a view of it, so resuming
        # re-renders the document from the stored numbers.  That costs no
        # rollout and no training, and it means a report can be brought up to
        # date with the reporting code without re-running an experiment - or,
        # worse, hand-editing a result file.
        logger.info("resuming: %s is already complete", paths.summary)
        summary = json.loads(paths.summary.read_text(encoding="utf-8"))
        paths.report.write_text(
            render_diagnostics_report(summary, decimals=settings.decimals),
            encoding="utf-8",
        )
        return summary

    spec = settings.run_spec(paths.summary)
    plan = run_plan(settings, spec, paths)
    logger.info(
        "diagnostics %s: agents from %s, %d evaluation iterations",
        settings.tag,
        plan["agents"],
        settings.eval_iterations,
    )
    if dry_run:
        logger.info("estimated training cost: %s", plan["estimated_training"])
        return {"tag": settings.tag, "dry_run": True, "plan": plan}

    stakeholders, source, origin, training = _resolve_agents(settings, spec, paths)
    env = build_environment(spec)
    log = collect_appraisals(
        env,
        stakeholders,
        investor_for(env, spec) if settings.investor else None,
        iterations=settings.eval_iterations,
        seed=settings.seed,
        qolu_reward=spec.qolu_reward_config(),
    )
    summary = diagnostics_summary(
        log,
        np.random.default_rng(settings.diagnostics_seed),
        headline_target=settings.headline_target,
    )
    summary.update(
        {
            "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "tag": settings.tag,
            "run": settings.as_record(),
            "plan": plan,
            "agents": {
                "source": origin,
                "directory": str(source),
                "n_checkpoints": len(agent_checkpoint_paths(source)),
                "investor": INVESTOR_STATE,
            },
            # Both halves of the run in one document: what the learner did
            # (``training``), and whether its target network was ever
            # synchronised at all (``learner``).  The second is reported
            # whichever way the agents arrived, because the counter is
            # restored from the checkpoint.
            "training": training,
            "learner": learner_record(stakeholders),
        }
    )

    paths.root.mkdir(parents=True, exist_ok=True)
    figure = calibration_figure(
        summary["pooled"]["targets"][settings.headline_target],
        paths.figures,
        f"distribution_calibration_{settings.tag}",
        source=str(paths.root),
        command=(
            "python -m chapter4_qolu.run_distribution_diagnostics --tag "
            f"{settings.tag}"
        ),
        target=settings.headline_target,
    )
    summary["figure"] = figure.as_record()
    paths.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths.report.write_text(
        render_diagnostics_report(summary, decimals=settings.decimals),
        encoding="utf-8",
    )
    logger.info("wrote %s", paths.summary)
    logger.info("wrote %s", paths.report)
    logger.info("verdict: %s", summary["verdict"]["statement"])
    return summary


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = settings_from(arguments)
    if is_sweep(arguments):
        run_sweep(
            settings,
            scenarios=arguments.scenarios or DEFAULT_SWEEP_SCENARIOS,
            seeds=(
                arguments.seeds if arguments.seeds is not None else DEFAULT_SWEEP_SEEDS
            ),
            parallel=max(1, int(arguments.parallel)),
            dry_run=arguments.dry_run,
            execute=not arguments.report_only,
        )
        return 0
    run_distribution_diagnostics(settings, dry_run=arguments.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
