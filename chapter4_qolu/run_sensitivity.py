"""Sweep the package's configurable readings against the published Tables 4-7.

The paper fixes neither the study extent, nor the per-iteration conversion
budget, nor which of three readings of ``R_QOLU,i`` the agents learn from, nor
any implementation detail of the three baselines.  Every one of those choices
moves the numbers.  This driver sweeps them, scores every combination against
the published ordering, and reports **all of it**, so the configuration it
selects can be read next to the whole distribution it came from.

Four stages::

    stage A  screening.  Every configuration on Scenario 1, seed 0, 1000
             iterations, the literal (unweighted) reading.  Scenario 1 weights
             the three stakeholders equally, so the literal and weighted
             readings coincide there.

    stage B  confirmation.  The top configurations over all four scenarios,
             three seeds and *both* consensus readings, scored on all 48
             published cells.

    stage B2 transfer.  The winning recipe, unchanged, on *every* window the
             Southeast scan retained plus the whole-region box - so the recipe
             can be told apart from the window stage A selected it on.

    stage C  deliverable.  The single best configuration at five seeds, with
             the full campaign documents and the figure set.

Usage::

    python -m chapter4_qolu.run_sensitivity --raster path/to/lcm_gb.img --dry-run
    python -m chapter4_qolu.run_sensitivity --raster ... --parallel 16 --resume
    python -m chapter4_qolu.run_sensitivity --report-only
    python -m chapter4_qolu.run_sensitivity --raster ... --stages a

Outputs land in ``results/sensitivity_<tag>/``: ``stageA/``, ``stageB/``,
``stageB2/`` and ``stageC/`` campaign folders in the ordinary
:mod:`chapter4_qolu.reporting.paths` layout, plus ``sensitivity_summary.json``,
``sensitivity.md`` and ``figures/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from chapter4_qolu.campaign_plan import format_duration, pending_runs
from chapter4_qolu.data.region_mask import region_mask_note
from chapter4_qolu.experiment import DEFAULT_RESULTS_DIR, resolve_raster
from chapter4_qolu.reporting.sensitivity_report import render_sensitivity_report
from chapter4_qolu.run_scenarios import execute_specs
from chapter4_qolu.scenario_worker import RunSpec
from chapter4_qolu.sensitivity_b2 import b2_environments
from chapter4_qolu.sensitivity_plan import (
    CAPS,
    DEFAULT_ENVIRONMENTS,
    PMADDPG_VARIANTS,
    REWARD_MODES,
    SensitivityConfiguration,
    SensitivityPaths,
    SensitivitySettings,
    estimate_stage_seconds,
    sensitivity_paths,
)
from chapter4_qolu.sensitivity_stages import (
    stage_a_runs,
    stage_b2_runs,
    stage_b_runs,
    stage_c_runs,
)
from chapter4_qolu.sensitivity_summary import (
    load_stage_a_index,
    published_record,
    reference_records,
    score_all_configurations,
    select_stage_b_choices,
    stage_a_record,
    stage_b2_records,
    stage_b2_windows_on_disk,
    stage_b_records,
    stage_c_record,
)

__all__ = [
    "SensitivityPaths",
    "SensitivitySettings",
    "main",
    "run_sensitivity",
    "sensitivity_paths",
]

logger = logging.getLogger(__name__)

#: Which stages ``--stages`` understands, in execution order.
STAGES: tuple[str, ...] = ("a", "b", "b2", "c")


def _execute(
    specs: Sequence[RunSpec], settings: SensitivitySettings, stage: str, dry_run: bool
) -> None:
    """Run one stage in a single pool, longest-first, honouring ``--resume``."""
    todo, done = pending_runs(specs, settings.resume)
    serial = estimate_stage_seconds(todo)
    workers = max(1, min(settings.parallel, max(len(todo), 1)))
    logger.info(
        "stage %s: %d runs (%d to run, %d already on disk)",
        stage.upper(),
        len(specs),
        len(todo),
        len(done),
    )
    logger.info(
        "stage %s: estimated serial cost %s; ETA at --parallel %d: %s",
        stage.upper(),
        format_duration(serial),
        workers,
        format_duration(serial / workers),
    )
    if not todo or dry_run:
        return
    outcomes = execute_specs(todo, settings.parallel)
    failures = [outcome for outcome in outcomes if not outcome["ok"]]
    if failures:
        logger.warning(
            "stage %s: %d of %d runs failed",
            stage.upper(),
            len(failures),
            len(outcomes),
        )


def _settings_record(settings: SensitivitySettings) -> dict[str, Any]:
    shapes = {window.label: list(window.shape) for window in settings.windows}
    return {
        "raster": Path(settings.raster).name,
        "environments": list(settings.environments),
        "grid_shapes": shapes,
        "grid_shape_text": ", ".join(
            f"`{label}` {rows} x {cols}" for label, (rows, cols) in shapes.items()
        ),
        "iterations": settings.iterations,
        "stage_a_seed": settings.stage_a_seed,
        "stage_b_seeds": list(settings.stage_b_seeds),
        "stage_c_seeds": list(settings.stage_c_seeds),
        "parallel": settings.parallel,
        "region_mask": (
            Path(settings.region_mask).name if settings.region_mask else None
        ),
        "region_mask_note": region_mask_note(settings.region_mask),
        "taxonomy_note": (
            "The land-cover taxonomy is **not** one of the swept axes: the "
            "paper's own land-preference legend uses 8 classes (Arable Land, "
            "Grassland, Heather Grassland, Littoral Sediment, Freshwater, "
            "Urban, Woodland, Suburban) derived from the LCM2015 *21 target "
            "classes*, while this package loads the *10 aggregate classes*. "
            "The difference is recorded here rather than swept."
        ),
        "reference_runs": [item.as_record() for item in settings.reference_runs],
    }


def run_sensitivity(
    settings: SensitivitySettings,
    dry_run: bool = False,
    stages: Sequence[str] = STAGES,
) -> dict[str, Any]:
    """Run the requested stages and write the summary and the report."""
    paths = sensitivity_paths(settings)
    paths.root.mkdir(parents=True, exist_ok=True)
    wanted = {stage.lower() for stage in stages}

    if "a" in wanted:
        _execute(stage_a_runs(settings), settings, "a", dry_run)
    index = load_stage_a_index(settings)
    scores = score_all_configurations(settings, index)
    choices = select_stage_b_choices(scores, settings.stage_b_configurations)

    if "b" in wanted and choices:
        _execute(stage_b_runs(settings, choices), settings, "b", dry_run)
    if dry_run:
        if "c" in wanted and choices:
            _execute(
                stage_c_runs(settings, choices[0].configuration, weighted=False),
                settings,
                "c",
                dry_run,
            )
        # A dry run prints the plan and the ETA; it must not overwrite the
        # documents of a sweep that has actually been performed.
        return {"tag": settings.tag, "settings": _settings_record(settings)}

    stage_b = stage_b_records(settings, choices) if choices else []
    winner = stage_b[0] if stage_b else None

    stage_b2: dict[str, Any] = {}
    deliverable: SensitivityConfiguration | None = None
    weighted = False
    if winner is not None:
        deliverable = next(
            item.configuration
            for item in choices
            if item.configuration.label == winner["configuration"]["label"]
        )
        weighted = bool(winner["weighted"])
    if deliverable is not None:
        # Executing stage B2 needs the raster (the scan grid is defined in BNG
        # kilometres); *reporting* it must not, so that --report-only can
        # rebuild every document from the records alone.  The windows that
        # were actually run are exactly the campaign directories on disk.
        if "b2" in wanted:
            windows = b2_environments(settings.raster)
            _execute(
                stage_b2_runs(settings, deliverable, windows, weighted),
                settings,
                "b2",
                dry_run,
            )
        else:
            windows = stage_b2_windows_on_disk(settings)
        entries = (
            stage_b2_records(settings, deliverable, windows, weighted)
            if windows
            else []
        )
        stage_b2 = {
            "recipe": {
                "label": deliverable.label,
                "description": deliverable.description,
                "reading": "weighted" if weighted else "plain",
            },
            "fitted_window": deliverable.environment,
            "n_windows": len(windows),
            "windows": entries,
        }
        # A window that reproduces the ordering under the winning recipe is a
        # better home for the deliverable than the window stage A selected it
        # on; without one, stage C stays where stage B left it and says so.
        qualifying = [entry for entry in entries if entry.get("qualifies")]
        if qualifying:
            deliverable = replace(
                deliverable, environment=str(qualifying[0]["window"]["label"])
            )
            stage_b2["chosen_window"] = qualifying[0]["window"]
        else:
            stage_b2["chosen_window"] = None

    stage_c: dict[str, Any] | None = None
    if deliverable is not None:
        if "c" in wanted:
            _execute(
                stage_c_runs(settings, deliverable, weighted), settings, "c", dry_run
            )
        # As for stage B2: *reporting* the deliverable reads whatever finished,
        # so --report-only rebuilds its tables without running anything.
        stage_c = stage_c_record(settings, deliverable, weighted)

    summary = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tag": settings.tag,
        "settings": _settings_record(settings),
        "published": published_record(),
        "stage_a": stage_a_record(settings, scores, index),
        "reference": reference_records(settings, index),
        "stage_b": {
            "seeds": list(settings.stage_b_seeds),
            "scenarios": list(settings.stage_b_scenarios),
            "choices": [choice.as_record() for choice in choices],
            "campaigns": stage_b,
        },
        "stage_b2": stage_b2,
        "stage_c": stage_c,
        "selection": {
            "n_configurations": len(settings.configurations()),
            "n_cells": len(scores),
            # The stage-A fallback has to be the *ranked* first configuration,
            # which is what `choices` holds; `scores` is in configuration order
            # and its first element means nothing.
            "winner": (
                winner["configuration"]
                if winner
                else (choices[0].configuration.as_record() if choices else None)
            ),
            "winner_reading": winner["reading"] if winner else None,
            "winner_stage": "B" if winner else ("A" if choices else None),
        },
    }

    paths.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths.report.write_text(render_sensitivity_report(summary), encoding="utf-8")
    logger.info("wrote %s", paths.summary)
    logger.info("wrote %s", paths.report)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", default=None, help="LCM raster (.img/.tif/.zip)")
    parser.add_argument("--tag", default=None, help="results folder suffix")
    parser.add_argument(
        "--environments",
        "--windows",
        dest="environments",
        nargs="+",
        default=None,
        metavar="LABEL",
        help=f"environments to sweep (default: {' '.join(DEFAULT_ENVIRONMENTS)})",
    )
    parser.add_argument(
        "--caps",
        nargs="+",
        type=int,
        default=None,
        help=f"per-iteration conversion caps (default: {' '.join(map(str, CAPS))})",
    )
    parser.add_argument("--reward-modes", nargs="+", default=None, choices=REWARD_MODES)
    parser.add_argument(
        "--pmaddpg-variants", nargs="+", default=None, choices=PMADDPG_VARIANTS
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--stage-b-configurations", type=int, default=None)
    parser.add_argument("--stage-b-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--stage-c-seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=list(STAGES),
        choices=STAGES,
        help="which stages to execute (the report always covers everything "
        "already on disk)",
    )
    parser.add_argument("--region-mask", default=None, metavar="PATH")
    parser.add_argument("--decimals", type=int, default=None, choices=(3, 4))
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild sensitivity_summary.json and sensitivity.md from disk",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and the ETA, then stop"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    return parser


def _settings_from(arguments: argparse.Namespace) -> SensitivitySettings:
    raster = (
        Path("(not loaded: --report-only)")
        if arguments.report_only and not arguments.raster
        else resolve_raster(arguments.raster)
    )
    defaults = SensitivitySettings(raster=raster, output_dir=Path(arguments.output_dir))
    return replace(
        defaults,
        tag=arguments.tag or defaults.tag,
        environments=(
            tuple(arguments.environments)
            if arguments.environments
            else defaults.environments
        ),
        caps=tuple(arguments.caps) if arguments.caps else defaults.caps,
        reward_modes=(
            tuple(arguments.reward_modes)
            if arguments.reward_modes
            else defaults.reward_modes
        ),
        pmaddpg_variants=(
            tuple(arguments.pmaddpg_variants)
            if arguments.pmaddpg_variants
            else defaults.pmaddpg_variants
        ),
        iterations=arguments.iterations or defaults.iterations,
        stage_b_configurations=(
            arguments.stage_b_configurations or defaults.stage_b_configurations
        ),
        stage_b_seeds=(
            tuple(arguments.stage_b_seeds)
            if arguments.stage_b_seeds
            else defaults.stage_b_seeds
        ),
        stage_c_seeds=(
            tuple(arguments.stage_c_seeds)
            if arguments.stage_c_seeds
            else defaults.stage_c_seeds
        ),
        region_mask=Path(arguments.region_mask) if arguments.region_mask else None,
        decimals=arguments.decimals or defaults.decimals,
        parallel=arguments.parallel or defaults.parallel,
        resume=arguments.resume or arguments.report_only,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = _settings_from(arguments)
    if arguments.report_only:
        # Everything is already on disk; --resume makes every stage a no-op.
        run_sensitivity(settings, stages=())
        return 0
    run_sensitivity(settings, dry_run=arguments.dry_run, stages=arguments.stages)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
