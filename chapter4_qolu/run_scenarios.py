"""The paper's four-scenario campaign: every method, every scenario, every seed.

``experiment.py`` runs one scenario of QOLU + BUIA and ``run_baseline`` runs one
baseline; this module runs the whole comparison and builds the result tables in
the layout of the paper's Tables 4-7.

Usage::

    # smoke test (4 methods, 50 iterations, one 30x30 cell; ~1 minute)
    python -m chapter4_qolu.run_scenarios --raster path/to/ukceh_download.zip --smoke

    # the published comparison: 4 methods x 4 scenarios x 5 seeds x 2 extents
    python -m chapter4_qolu.run_scenarios \\
        --raster path/to/ukceh_download.zip \\
        --tag main --seeds 0 1 2 3 4 --parallel 16

    # the same comparison with the scenario weights reaching the top-down
    # agents (both switches are off by default; the tag keeps the runs apart)
    python -m chapter4_qolu.run_scenarios \\
        --raster path/to/ukceh_download.zip \\
        --tag weighted --scale-rewards-by-weight --weighted-consensus

    # rebuild summary.json / tables.md / tables.tex / pattern_check.md only
    python -m chapter4_qolu.run_scenarios --tag main --aggregate

Results land in ``results/campaign_<tag>/<configuration>/scenario<k>/
<method>_seed<N>.json``; ``--resume`` skips any run whose record is already
there.  Every method inside a (configuration, scenario, seed) cell sees the same
raster crop, the same environment settings and the same seed - see
:mod:`chapter4_qolu.scenario_worker`.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from chapter4_qolu.campaign_plan import (
    DEFAULT_CONFIGS,
    DEFAULT_METHODS,
    DEFAULT_SCENARIOS,
    DEFAULT_SEEDS,
    METHOD_COST_SECONDS,
    CampaignSettings,
    campaign_runs,
    estimate_runtime_seconds,
    format_duration,
    pending_runs,
    smoke_settings,
)
from chapter4_qolu.config import SCENARIOS
from chapter4_qolu.experiment import DEFAULT_RESULTS_DIR, resolve_raster
from chapter4_qolu.reporting.aggregate import load_run_records
from chapter4_qolu.reporting.derived import (
    MISSING,
    MissingLandCoverGrids,
    metric_source,
    missing_grids_message,
)
from chapter4_qolu.reporting.documents import build_summary, write_campaign_documents
from chapter4_qolu.reporting.paths import parse_scale_config
from chapter4_qolu.scenario_worker import METHOD_KEYS, RunSpec, execute_run
from chapter4_qolu.window_scan import SOUTH_EAST_REGION_LABEL, parse_window_label

__all__ = [
    "DEFAULT_CONFIGS",
    "DEFAULT_METHODS",
    "DEFAULT_SCENARIOS",
    "DEFAULT_SEEDS",
    "METHOD_COST_SECONDS",
    "METHOD_KEYS",
    "CampaignSettings",
    "aggregate_campaign",
    "campaign_runs",
    "estimate_runtime_seconds",
    "execute_specs",
    "format_duration",
    "main",
    "pending_runs",
    "run_campaign",
    "smoke_settings",
]

logger = logging.getLogger(__name__)


def _log_plan(
    settings: CampaignSettings,
    specs: Sequence[RunSpec],
    todo: Sequence[RunSpec],
    done: Sequence[RunSpec],
) -> None:
    serial = estimate_runtime_seconds(todo)
    workers = max(1, min(settings.parallel, max(len(todo), 1)))
    logger.info("campaign %r -> %s", settings.tag, settings.campaign_dir)
    logger.info("  methods    : %s", ", ".join(settings.methods))
    logger.info("  scenarios  : %s", ", ".join(str(k) for k in settings.scenarios))
    logger.info("  seeds      : %s", ", ".join(str(s) for s in settings.seeds))
    logger.info(
        "  configs    : %s", ", ".join(scale.label for scale in settings.configs)
    )
    if settings.window_label:
        logger.info(
            "  window     : %s (BNG top-left %s km)",
            settings.window_label,
            settings.window_bng_km,
        )
    if settings.scale_by_scenario_weight or settings.weighted_consensus:
        # Only reported when something is on, so a default campaign's log
        # carries no such line and a weighted one cannot be mistaken for it.
        logger.info(
            "  weighting  : scale_by_scenario_weight=%s, weighted_consensus=%s",
            settings.scale_by_scenario_weight,
            settings.weighted_consensus,
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
    logger.info(
        "  (costs are measured 1000-iteration CPU runs at 50x50 with torch "
        "using every core; each worker here is pinned to one thread, so the "
        "real wall time will be longer)"
    )


def _worker(spec: RunSpec) -> dict[str, Any]:
    """Pool entry point: run one cell, never raise into the parent."""
    try:
        execute_run(spec)
    except Exception as error:  # noqa: BLE001 - one bad cell must not stop 160
        return {"path": str(spec.output_path), "ok": False, "error": repr(error)}
    return {"path": str(spec.output_path), "ok": True, "error": None}


def _execute(specs: Sequence[RunSpec], parallel: int) -> list[dict[str, Any]]:
    """Run every spec, in-process when ``parallel`` is 1, else in a pool.

    Design note: the pool is fed longest-first (the estimated cost of each run),
    which is the standard LPT heuristic - it keeps the expensive QOLU + BUIA
    runs from being scheduled last and stretching the makespan.
    """
    ordered = sorted(specs, key=lambda spec: -estimate_runtime_seconds([spec]))
    total = len(ordered)
    started = time.perf_counter()
    outcomes: list[dict[str, Any]] = []

    def _report(index: int, outcome: dict[str, Any]) -> None:
        elapsed = time.perf_counter() - started
        level = logging.INFO if outcome["ok"] else logging.ERROR
        logger.log(
            level,
            "[%d/%d] %s %s (elapsed %s)",
            index,
            total,
            "ok  " if outcome["ok"] else "FAIL",
            "/".join(Path(outcome["path"]).parts[-3:]),
            format_duration(elapsed),
        )
        if not outcome["ok"]:
            logger.error("        %s", outcome["error"])

    if max(1, int(parallel)) == 1 or total <= 1:
        for index, spec in enumerate(ordered, start=1):
            outcome = _worker(spec)
            outcomes.append(outcome)
            _report(index, outcome)
        return outcomes

    workers = min(int(parallel), total)
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        for index, outcome in enumerate(pool.imap_unordered(_worker, ordered), start=1):
            outcomes.append(outcome)
            _report(index, outcome)
    return outcomes


#: Public name of :func:`_execute`.  The window scan feeds *one* pool with the
#: runs of every window at once, so it needs the executor without the
#: single-campaign wrapper around it.
execute_specs = _execute


def _require_derivable_metrics(campaign_dir: Path) -> None:
    """Refuse to aggregate records that lack a metric and the grids to score it.

    Design note (published records): a run that did not record ``c_align``
    is scored on it from its stored land-cover grids.  The records published
    with this package carry no grids, so for those runs the documents could
    only be rebuilt without ``c_align``; stop and say how to regenerate the
    grids rather than silently write a different ``summary.json``.
    """
    lacking = [
        record
        for record in load_run_records(campaign_dir)
        if metric_source(record) == MISSING
    ]
    if lacking:
        raise MissingLandCoverGrids(
            missing_grids_message(
                "Rebuilding the campaign documents (c_align of the runs that "
                "did not record it)",
                campaign_dir,
                len(lacking),
            )
        )


def aggregate_campaign(
    settings: CampaignSettings, aggregate_only: bool = False
) -> dict[str, Any]:
    """Aggregate the campaign's records and write its four documents."""
    metadata = settings.metadata()
    if aggregate_only:
        # --aggregate reads whatever is on disk, which need not be what this
        # invocation asked for; say so rather than let the header imply it.
        metadata["note"] = (
            "rebuilt with --aggregate: the methods / scenarios / seeds listed "
            "here describe the invocation, not necessarily the records found"
        )
    _require_derivable_metrics(settings.campaign_dir)
    summary = build_summary(settings.campaign_dir, metadata=metadata)
    written = write_campaign_documents(
        settings.campaign_dir, summary, decimals=settings.decimals
    )
    for path in written.values():
        logger.info("wrote %s", path)
    return summary


def run_campaign(settings: CampaignSettings) -> dict[str, Any]:
    """Run every pending cell of the campaign and aggregate the results."""
    specs = campaign_runs(settings)
    todo, done = pending_runs(specs, settings.resume)
    _log_plan(settings, specs, todo, done)
    if todo:
        outcomes = _execute(todo, settings.parallel)
        failures = [outcome for outcome in outcomes if not outcome["ok"]]
        if failures:
            logger.warning(
                "%d of %d runs failed; aggregating the %d that finished",
                len(failures),
                len(outcomes),
                len(outcomes) - len(failures),
            )
    return aggregate_campaign(settings)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", default=None, help="LCM GeoTIFF or UKCEH zip")
    parser.add_argument("--tag", default=None, help="results folder name")
    parser.add_argument("--methods", nargs="+", default=None, choices=METHOD_KEYS)
    parser.add_argument(
        "--scenarios", nargs="+", type=int, default=None, choices=sorted(SCENARIOS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        metavar="GRID:FRACTION:CAP",
        help="scale configurations, e.g. 50:1.0:10 100:0.25:none",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--window",
        default=None,
        metavar="LABEL",
        help="pin every run to one scan window, e.g. E495_N140 (the south-west "
        "corner in British National Grid km) or SE_region for the whole "
        "South East box; without it the loader keeps its automatic "
        "densest-land crop",
    )
    parser.add_argument(
        "--region",
        action="store_true",
        help=f"shorthand for --window {SOUTH_EAST_REGION_LABEL}: the whole "
        "South East England bounding box, BNG E 410-650 km / N 60-270 km "
        "(240 x 210 parcels).  Pair it with --configs 210x240:1.0:<cap>",
    )
    parser.add_argument(
        "--region-mask",
        default=None,
        metavar="PATH",
        help="polygon boundary (GeoJSON / shapefile) clipping the window; "
        "without it the bounding box is used and every document says so",
    )
    parser.add_argument("--stakeholders", type=int, default=3)
    parser.add_argument(
        "--qolu-reward-mode",
        default="per_parcel",
        choices=("per_parcel", "objective", "team"),
    )
    parser.add_argument(
        "--scale-rewards-by-weight",
        action="store_true",
        help="pay stakeholder i n * w_i times its own per-parcel reward, i.e. "
        "the per-parcel reading of R = sum_i w_i R_phi_i (off by default)",
    )
    parser.add_argument(
        "--weighted-consensus",
        action="store_true",
        help="weight each stakeholder's contribution to the consensus sum by "
        "its scenario weight (off by default)",
    )
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=10.0,
        help="physical distance at which proximity saturates in view (b)",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=None,
        choices=(3, 4),
        help="decimal places in tables.md / tables.tex (default 3); "
        "summary.json and per_seed.csv always carry full precision",
    )
    parser.add_argument("--parallel", type=int, default=None, help="worker processes")
    parser.add_argument("--resume", action="store_true", help="skip finished runs")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="only rebuild the documents from existing records",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and the ETA, then stop"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    return parser


def _resolve_raster(arguments: argparse.Namespace) -> Path:
    """The raster to run on; aggregation never loads one."""
    if arguments.aggregate and not arguments.raster:
        return Path("(not loaded: --aggregate)")
    return resolve_raster(arguments.raster)


def _settings_from(arguments: argparse.Namespace, raster: Path) -> CampaignSettings:
    """Fold the CLI into :class:`CampaignSettings`, honouring ``--smoke``."""
    defaults = (
        smoke_settings(raster, Path(arguments.output_dir))
        if arguments.smoke
        else CampaignSettings(raster=raster, output_dir=Path(arguments.output_dir))
    )
    configs = (
        tuple(parse_scale_config(text) for text in arguments.configs)
        if arguments.configs
        else defaults.configs
    )
    # Design note (--window size): the window is as many kilometres across as the
    # first configuration has parcels, because a parcel of these products is
    # exactly 1 km^2.  A campaign that mixes extents therefore pins the *corner*
    # rather than the extent, which is what the window scan does too.
    label = getattr(arguments, "window", None)
    if getattr(arguments, "region", False):
        label = label or SOUTH_EAST_REGION_LABEL
    window = parse_window_label(label, size_km=configs[0].width) if label else None
    return replace(
        defaults,
        tag=arguments.tag or defaults.tag,
        methods=tuple(arguments.methods) if arguments.methods else defaults.methods,
        scenarios=(
            tuple(arguments.scenarios) if arguments.scenarios else defaults.scenarios
        ),
        seeds=tuple(arguments.seeds) if arguments.seeds else defaults.seeds,
        configs=configs,
        iterations=(
            defaults.iterations
            if arguments.iterations is None
            else arguments.iterations
        ),
        stakeholders=arguments.stakeholders,
        qolu_reward_mode=arguments.qolu_reward_mode,
        scale_by_scenario_weight=arguments.scale_rewards_by_weight,
        weighted_consensus=arguments.weighted_consensus,
        max_distance_km=arguments.max_distance_km,
        decimals=arguments.decimals or defaults.decimals,
        parallel=arguments.parallel or defaults.parallel,
        resume=arguments.resume,
        window_label=window.label if window else defaults.window_label,
        window_bng_km=(
            tuple(float(value) for value in window.bng_top_left_km)
            if window
            else defaults.window_bng_km
        ),
        region_mask=(
            Path(arguments.region_mask)
            if getattr(arguments, "region_mask", None)
            else defaults.region_mask
        ),
    )


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = _settings_from(arguments, _resolve_raster(arguments))

    if arguments.dry_run:
        specs = campaign_runs(settings)
        todo, done = pending_runs(specs, settings.resume)
        _log_plan(settings, specs, todo, done)
        return 0
    try:
        if arguments.aggregate:
            aggregate_campaign(settings, aggregate_only=True)
            return 0
        summary = run_campaign(settings)
    except MissingLandCoverGrids as error:
        raise SystemExit(f"error: {error}") from None
    logger.info(
        "aggregated %d runs into %s",
        summary["n_runs"],
        settings.campaign_dir / "summary.json",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
