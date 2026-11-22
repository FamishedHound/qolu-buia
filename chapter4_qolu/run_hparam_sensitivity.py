"""The +/-20 % hyper-parameter sensitivity analysis Chapter 1 describes.

Chapter 1 states that "sensitivity analyses are conducted by perturbing key
hyperparameters (typically +/-20%) to assess the robustness of comparative
conclusions".  This driver performs that analysis for Chapter 4: it
runs the configuration under test once as a baseline, then moves each key
hyper-parameter by +/-20 % **one at a time**, re-runs the comparison, and
reports whether the *method ranking* - the comparative conclusion the sentence
is about - is preserved, together with Kendall's tau against the baseline
ordering.

Usage::

    # the plan and the ETA, run nothing
    python -m chapter4_qolu.run_hparam_sensitivity --raster <raster> --dry-run

    # the analysis
    python -m chapter4_qolu.run_hparam_sensitivity --raster <raster> \\
        --tag ch4 --config 50:1.0:10 --window E495_N140 \\
        --seeds 0 1 2 --parallel 16 --resume

    # one hyper-parameter, moved by 10 % instead
    python -m chapter4_qolu.run_hparam_sensitivity --raster <raster> \\
        --hyperparameters learning_rate --fraction 0.1 --resume

    # rebuild the documents from the records on disk, run nothing
    python -m chapter4_qolu.run_hparam_sensitivity --tag ch4 --report-only

Output lands in ``results/hparam_<tag>/``: one ordinary campaign folder per
perturbation under ``runs/`` (so ``--resume``, the record layout and the
campaign documents are the machinery every other driver uses), plus
``hparam_summary.json``, ``hparam_sensitivity.md`` and ``hparam_runs.csv``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from chapter4_qolu.campaign_plan import format_duration, pending_runs
from chapter4_qolu.config import SCENARIOS
from chapter4_qolu.experiment import DEFAULT_RESULTS_DIR, resolve_raster
from chapter4_qolu.hparam_plan import (
    HParamPaths,
    HParamSettings,
    all_runs,
    effective_perturbations,
    estimate_seconds,
    hparam_paths,
    settings_perturbations,
)
from chapter4_qolu.hparam_sensitivity import PERTURBATION_FRACTION, hyperparameter_names
from chapter4_qolu.hparam_summary import build_hparam_summary, every_record
from chapter4_qolu.hparam_worker import execute_hparam_specs
from chapter4_qolu.methods import METHOD_KEYS
from chapter4_qolu.reporting.hparam_report import (
    render_hparam_report,
    render_hparam_runs_csv,
)
from chapter4_qolu.reporting.paths import parse_scale_config

__all__ = [
    "HParamPaths",
    "HParamSettings",
    "hparam_paths",
    "main",
    "run_hparam_sensitivity",
]

logger = logging.getLogger(__name__)


def _log_plan(settings: HParamSettings) -> None:
    specs = all_runs(settings)
    todo, done = pending_runs(specs, settings.resume)
    listed = settings_perturbations(settings)
    effective = effective_perturbations(settings)
    serial = estimate_seconds(todo)
    workers = max(1, min(settings.parallel, max(len(todo), 1)))
    logger.info(
        "hyper-parameter sensitivity %r -> %s",
        settings.tag,
        hparam_paths(settings).root,
    )
    logger.info("  configuration : %s", settings.scale.description)
    logger.info(
        "  window        : %s", settings.window or "automatic densest-land crop"
    )
    logger.info("  methods       : %s", ", ".join(settings.methods))
    logger.info("  scenarios     : %s", ", ".join(str(k) for k in settings.scenarios))
    logger.info("  seeds         : %s", ", ".join(str(s) for s in settings.seeds))
    logger.info(
        "  perturbations : %d listed, %d effective (%d are the identity after "
        "rounding or clamping)",
        len(listed),
        len(effective),
        len(listed) - len(effective),
    )
    for perturbation in listed:
        logger.info(
            "      %-34s %s%s",
            perturbation.label,
            perturbation.description,
            " [not run: identity]" if perturbation.is_no_op else "",
        )
    logger.info(
        "  %d runs of %d iterations (%d to run, %d already on disk)",
        len(specs),
        settings.iterations,
        len(todo),
        len(done),
    )
    logger.info(
        "  estimated serial cost %s; ETA at --parallel %d: %s",
        format_duration(serial),
        workers,
        format_duration(serial / workers),
    )


def _write(paths: HParamPaths, summary: dict[str, Any], records: list[Any]) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths.report.write_text(render_hparam_report(summary), encoding="utf-8")
    paths.runs_csv.write_text(render_hparam_runs_csv(records), encoding="utf-8")
    for path in (paths.summary, paths.report, paths.runs_csv):
        logger.info("wrote %s", path)


def run_hparam_sensitivity(
    settings: HParamSettings, dry_run: bool = False, execute: bool = True
) -> dict[str, Any]:
    """Run the analysis (or just report it) and write its three documents."""
    _log_plan(settings)
    if dry_run:
        # A dry run prints the plan and the ETA; it must not create a results
        # folder, let alone overwrite the documents of a finished analysis.
        return {"tag": settings.tag, "settings": settings.as_record()}

    if execute:
        todo, _ = pending_runs(all_runs(settings), settings.resume)
        if todo:
            outcomes = execute_hparam_specs(list(todo), settings.parallel)
            failures = [outcome for outcome in outcomes if not outcome["ok"]]
            if failures:
                logger.warning(
                    "%d of %d runs failed; reporting the %d that finished",
                    len(failures),
                    len(outcomes),
                    len(outcomes) - len(failures),
                )

    summary = build_hparam_summary(settings)
    _write(hparam_paths(settings), summary, every_record(settings))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", default=None, help="LCM .img / .tif / UKCEH .zip")
    parser.add_argument("--tag", default=None, help="results folder suffix")
    parser.add_argument(
        "--hyperparameters",
        nargs="+",
        default=None,
        choices=hyperparameter_names(),
        metavar="NAME",
        help="which hyper-parameters to perturb (default: all of "
        + ", ".join(hyperparameter_names())
        + ")",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help=f"relative perturbation (default {PERTURBATION_FRACTION}, the "
        "thesis's +/-20 %%)",
    )
    parser.add_argument("--methods", nargs="+", default=None, choices=METHOD_KEYS)
    parser.add_argument(
        "--scenarios", nargs="+", type=int, default=None, choices=sorted(SCENARIOS)
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="at least three, so a mean +/- SE and a paired t-test are defined",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--config",
        default=None,
        metavar="GRID:FRACTION:CAP",
        help="the configuration under test, e.g. 50:1.0:10",
    )
    parser.add_argument(
        "--window",
        default=None,
        metavar="LABEL",
        help="pin every run to one window, e.g. E495_N140 or SE_region",
    )
    parser.add_argument("--region-mask", default=None, metavar="PATH")
    parser.add_argument("--stakeholders", type=int, default=None)
    parser.add_argument(
        "--qolu-reward-mode", default=None, choices=("per_parcel", "objective", "team")
    )
    parser.add_argument("--max-distance-km", type=float, default=None)
    parser.add_argument("--decimals", type=int, default=None, choices=(3, 4))
    parser.add_argument("--parallel", type=int, default=None, help="worker processes")
    parser.add_argument("--resume", action="store_true", help="skip finished runs")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild hparam_summary.json, hparam_sensitivity.md and "
        "hparam_runs.csv from the records on disk",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and the ETA, then stop"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    return parser


def _settings_from(arguments: argparse.Namespace) -> HParamSettings:
    raster = (
        Path("(not loaded: --report-only)")
        if arguments.report_only and not arguments.raster
        else resolve_raster(arguments.raster)
    )
    defaults = HParamSettings(raster=raster, output_dir=Path(arguments.output_dir))
    return replace(
        defaults,
        tag=arguments.tag or defaults.tag,
        hyperparameters=(
            tuple(arguments.hyperparameters) if arguments.hyperparameters else None
        ),
        fraction=(
            defaults.fraction if arguments.fraction is None else arguments.fraction
        ),
        methods=tuple(arguments.methods) if arguments.methods else defaults.methods,
        scenarios=(
            tuple(arguments.scenarios) if arguments.scenarios else defaults.scenarios
        ),
        seeds=tuple(arguments.seeds) if arguments.seeds else defaults.seeds,
        iterations=arguments.iterations or defaults.iterations,
        scale=(
            parse_scale_config(arguments.config) if arguments.config else defaults.scale
        ),
        stakeholders=arguments.stakeholders or defaults.stakeholders,
        qolu_reward_mode=arguments.qolu_reward_mode or defaults.qolu_reward_mode,
        max_distance_km=arguments.max_distance_km or defaults.max_distance_km,
        window=arguments.window or defaults.window,
        region_mask=Path(arguments.region_mask) if arguments.region_mask else None,
        decimals=arguments.decimals or defaults.decimals,
        parallel=arguments.parallel or defaults.parallel,
        resume=arguments.resume or arguments.report_only,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = _settings_from(arguments)
    run_hparam_sensitivity(
        settings, dry_run=arguments.dry_run, execute=not arguments.report_only
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
