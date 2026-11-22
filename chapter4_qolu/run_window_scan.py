"""Scan Southeast England for the window closest to the published Tables 4-7.

The paper says "the Southeast UK region, 2015" and a 50 x 50 grid of 1 km^2
parcels but never states an extent.  Rather than pick a window and present it
as *the* study region, this driver scans a grid of candidate windows and
reports the whole distribution, so the selection rule and the number of
candidates it ranged over are both visible.

Two stages::

    stage 1  every retained window, all four methods, Scenario 1, seed 0,
             1000 iterations, configuration 50:1.0:10, --qolu-reward-mode
             per_parcel, the *literal* (unweighted) reading.  Scenario 1 has
             equal weights, so the literal and weighted readings coincide there
             and the screening does not depend on which one is preferred.

    stage 2  the top three windows, all four scenarios x three seeds x four
             methods, the *weighted* reading (``--scale-rewards-by-weight
             --weighted-consensus``) - the only reading in which the scenario
             weights reach QOLU + BUIA and P-MADDPG at all.

Usage::

    python -m chapter4_qolu.run_window_scan --raster path/to/lcm_gb.img
    python -m chapter4_qolu.run_window_scan --raster ... --dry-run
    python -m chapter4_qolu.run_window_scan --raster ... --resume
    python -m chapter4_qolu.run_window_scan --report-only

Outputs land in ``results/window_scan_<tag>/``: ``stage1/`` and ``stage2/``
campaign folders (the ordinary layout of
:mod:`chapter4_qolu.reporting.paths`), ``scan_summary.json`` and ``scan.md``.
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

from chapter4_qolu.campaign_plan import (
    CampaignSettings,
    campaign_runs,
    estimate_runtime_seconds,
    format_duration,
    pending_runs,
)
from chapter4_qolu.data.lcm_loader import load_lcm_raster
from chapter4_qolu.experiment import DEFAULT_RESULTS_DIR, resolve_raster
from chapter4_qolu.reporting.paths import parse_scale_config
from chapter4_qolu.reporting.scan_report import render_scan_report
from chapter4_qolu.run_scenarios import execute_specs
from chapter4_qolu.scan_plan import (
    PUBLISHED_CONFIG,
    SINGLE_THREAD_COST_FACTOR,
    ScanPaths,
    ScanSettings,
    estimate_stage_seconds,
    scan_paths,
    stage1_settings,
    stage2_settings,
    stage_campaigns,
)
from chapter4_qolu.scan_summary import (
    grid_record,
    metric_values_from_records,
    raster_record,
    score_distribution,
    stage1_window_records,
    stage2_window_records,
)
from chapter4_qolu.scenario_worker import RunSpec
from chapter4_qolu.window_scan import ScanWindow, candidate_windows, retain_windows

__all__ = [
    "PUBLISHED_CONFIG",
    "ScanPaths",
    "ScanSettings",
    "estimate_stage_seconds",
    "main",
    "metric_values_from_records",
    "run_scan",
    "scan_paths",
    "stage1_settings",
    "stage2_settings",
]

logger = logging.getLogger(__name__)


def _run_stage(
    settings: ScanSettings,
    windows: Sequence[ScanWindow],
    stage: int,
    dry_run: bool = False,
) -> list[CampaignSettings]:
    """Execute every run of one stage in a single pool, longest-first."""
    campaigns = stage_campaigns(settings, windows, stage)
    specs: list[RunSpec] = []
    for campaign in campaigns:
        specs.extend(campaign_runs(campaign))
    todo, done = pending_runs(specs, settings.resume)
    # The same single-thread correction estimate_stage_seconds applies, but on
    # the runs that are actually still to do rather than on the whole stage.
    serial = estimate_runtime_seconds(todo) * SINGLE_THREAD_COST_FACTOR
    workers = max(1, min(settings.parallel, max(len(todo), 1)))
    logger.info(
        "stage %d: %d windows, %d runs (%d to run, %d already on disk)",
        stage,
        len(windows),
        len(specs),
        len(todo),
        len(done),
    )
    logger.info(
        "stage %d: estimated serial cost %s; ETA at --parallel %d: %s",
        stage,
        format_duration(serial),
        workers,
        format_duration(serial / workers),
    )
    if todo and not dry_run:
        outcomes = execute_specs(todo, settings.parallel)
        failures = [outcome for outcome in outcomes if not outcome["ok"]]
        if failures:
            logger.warning(
                "stage %d: %d of %d runs failed", stage, len(failures), len(outcomes)
            )
    return campaigns


def run_scan(settings: ScanSettings, dry_run: bool = False) -> dict[str, Any]:
    """Run both stages and write ``scan_summary.json`` and ``scan.md``."""
    paths = scan_paths(settings)
    paths.root.mkdir(parents=True, exist_ok=True)

    raster = load_lcm_raster(settings.raster)
    windows = candidate_windows(
        settings.box, settings.window_size_km, settings.stride_km
    )
    kept, dropped = retain_windows(raster, windows, settings.max_sea_fraction)
    logger.info(
        "grid: %s, %d x %d km windows at %d km stride -> %d candidates, "
        "%d retained at <= %.0f%% sea",
        settings.box.description,
        settings.window_size_km,
        settings.window_size_km,
        settings.stride_km,
        len(windows),
        len(kept),
        settings.max_sea_fraction * 100.0,
    )

    retained_windows = [item.window for item in kept]
    _run_stage(settings, retained_windows, stage=1, dry_run=dry_run)
    scored = stage1_window_records(settings, kept)

    top = [
        ScanWindow(
            easting_km=int(entry["window"]["easting_km"]),
            northing_km=int(entry["window"]["northing_km"]),
            size_km=int(entry["window"]["size_km"]),
        )
        for entry in scored[: settings.stage2_windows]
    ]
    _run_stage(settings, top, stage=2, dry_run=dry_run)
    if dry_run:
        # A dry run has printed the plan and the ETA; it must not overwrite the
        # documents of a scan that has actually been performed.
        return {"grid": grid_record(settings, len(windows), len(kept))}
    stage2 = stage2_window_records(settings, top)

    summary = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tag": settings.tag,
        "raster": raster_record(raster),
        "grid": grid_record(settings, len(windows), len(kept)),
        "dropped": [item.as_record() for item in dropped],
        "stage1": {
            "iterations": settings.iterations,
            "configuration": settings.config.label,
            "scenario": 1,
            "seeds": [0],
            "methods": list(settings.methods),
            "weighted": False,
            "qolu_reward_mode": settings.qolu_reward_mode,
            "windows": scored,
            "score_distribution": score_distribution(scored),
        },
        "stage2": {
            "iterations": settings.iterations,
            "configuration": settings.config.label,
            "seeds": list(settings.stage2_seeds),
            "scenarios": [1, 2, 3, 4],
            "weighted": True,
            "windows": stage2,
            "winner": stage2[0]["window"] if stage2 else None,
        },
        "selection": {
            "n_candidate_windows": len(windows),
            "n_scanned_windows": len(kept),
            "n_confirmed_windows": len(top),
            "winner": (
                stage2[0]["window"]
                if stage2
                else (scored[0]["window"] if scored else None)
            ),
        },
    }

    paths.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths.report.write_text(render_scan_report(summary), encoding="utf-8")
    logger.info("wrote %s", paths.summary)
    logger.info("wrote %s", paths.report)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", default=None, help="LCM raster (.img/.tif/.zip)")
    parser.add_argument("--tag", default=None, help="results folder suffix")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--config",
        default=None,
        metavar="GRID:FRACTION:CAP",
        help="grid:fraction:cap, e.g. 50:1.0:1 (default 50:1.0:10)",
    )
    parser.add_argument("--stage2-windows", type=int, default=None)
    parser.add_argument("--stage2-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-sea-fraction", type=float, default=None)
    parser.add_argument("--stride-km", type=int, default=None)
    parser.add_argument("--decimals", type=int, default=None, choices=(3, 4))
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild scan_summary.json and scan.md from the records on disk",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and the ETA, then stop"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    return parser


def _settings_from(arguments: argparse.Namespace) -> ScanSettings:
    raster = (
        Path("(not loaded: --report-only)")
        if arguments.report_only and not arguments.raster
        else resolve_raster(arguments.raster)
    )
    defaults = ScanSettings(raster=raster, output_dir=Path(arguments.output_dir))
    return replace(
        defaults,
        tag=arguments.tag or defaults.tag,
        iterations=arguments.iterations or defaults.iterations,
        config=(
            parse_scale_config(arguments.config)
            if arguments.config
            else defaults.config
        ),
        stage2_windows=arguments.stage2_windows or defaults.stage2_windows,
        stage2_seeds=(
            tuple(arguments.stage2_seeds)
            if arguments.stage2_seeds
            else defaults.stage2_seeds
        ),
        max_sea_fraction=(
            defaults.max_sea_fraction
            if arguments.max_sea_fraction is None
            else arguments.max_sea_fraction
        ),
        stride_km=arguments.stride_km or defaults.stride_km,
        decimals=arguments.decimals or defaults.decimals,
        parallel=arguments.parallel or defaults.parallel,
        resume=arguments.resume or arguments.report_only,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = _settings_from(arguments)
    if arguments.report_only:
        # Everything is already on disk; --resume makes both stages no-ops.
        run_scan(settings)
        return 0
    run_scan(settings, dry_run=arguments.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
