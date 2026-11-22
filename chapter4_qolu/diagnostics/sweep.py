"""Run the return-distribution diagnostics over scenarios and seeds.

A single diagnostics run is one seed of one scenario.  A claim such as "the
learnt return distribution carries (or does not carry) state-conditional
information" is a claim about the *method*, and one seed cannot separate the
method from the seed.  This module runs one ordinary
diagnostics run per scenario x seed cell, each with its own tag, its own
folder and its own documents, and then aggregates across them.

Design note (each cell is a whole diagnostics run, not a shortcut): a cell
writes the same ``distribution_diagnostics.json``, ``.md``, ``agents/`` and
``figures/`` a lone ``--scenario N --seed M`` invocation writes, in
``<root>/runs/distribution_diagnostics_<tag>_s<N>_seed<M>/``.  Nothing in the
sweep recomputes a statistic: the aggregation reads the cells back off disk, so
a sweep table and the per-cell document it points at cannot disagree, and
``--report-only`` rebuilds everything without running anything.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import multiprocessing
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from chapter4_qolu.campaign_plan import estimate_runtime_seconds, format_duration
from chapter4_qolu.diagnostics.plan import (
    DIAGNOSTICS_PREFIX,
    DiagnosticsSettings,
    diagnostics_paths,
)
from chapter4_qolu.diagnostics.returns import TARGET_NAMES
from chapter4_qolu.diagnostics.sweep_report import render_sweep_report
from chapter4_qolu.diagnostics.sweep_summary import (
    aggregate_records,
    cell_record,
    per_objective_records,
    per_scenario_records,
    sweep_verdict,
)

__all__ = [
    "DEFAULT_SWEEP_SCENARIOS",
    "DEFAULT_SWEEP_SEEDS",
    "SWEEP_STEM",
    "SweepPaths",
    "cell_settings",
    "run_sweep",
    "sweep_cells",
    "sweep_paths",
]

logger = logging.getLogger(__name__)

#: The four published scenario weightings.
DEFAULT_SWEEP_SCENARIOS: tuple[int, ...] = (1, 2, 3, 4)
#: Design note: three seeds, the same floor ``reporting/stats.py`` puts on a
#: paired test.  Two cells give a standard error made of one difference.
DEFAULT_SWEEP_SEEDS: tuple[int, ...] = (0, 1, 2)
#: Stem of the two sweep documents, so the pair is obviously one thing.
SWEEP_STEM: str = "distribution_diagnostics_sweep"


@dataclass(frozen=True)
class SweepPaths:
    """Where one sweep writes: the cells below it, the two documents beside."""

    root: Path
    runs: Path
    summary: Path
    report: Path


def sweep_paths(settings: DiagnosticsSettings) -> SweepPaths:
    """Result-file layout of one sweep."""
    root = Path(settings.output_dir) / f"{DIAGNOSTICS_PREFIX}{settings.tag}"
    return SweepPaths(
        root=root,
        runs=root / "runs",
        summary=root / f"{SWEEP_STEM}.json",
        report=root / f"{SWEEP_STEM}.md",
    )


def cell_settings(
    base: DiagnosticsSettings, scenario: int, seed: int
) -> DiagnosticsSettings:
    """One cell of the sweep, described exactly as a lone run would be."""
    return replace(
        base,
        scenario=int(scenario),
        seed=int(seed),
        tag=f"{base.tag}_s{int(scenario)}_seed{int(seed)}",
        output_dir=sweep_paths(base).runs,
    )


def sweep_cells(
    base: DiagnosticsSettings,
    scenarios: Sequence[int] = DEFAULT_SWEEP_SCENARIOS,
    seeds: Sequence[int] = DEFAULT_SWEEP_SEEDS,
) -> tuple[DiagnosticsSettings, ...]:
    """Every cell, scenario-major so a partial sweep covers whole scenarios."""
    if not scenarios or not seeds:
        raise ValueError("a sweep needs at least one scenario and one seed")
    return tuple(
        cell_settings(base, scenario, seed) for scenario in scenarios for seed in seeds
    )


def _worker(settings: DiagnosticsSettings) -> dict[str, Any]:
    """Pool entry point: run one cell, never raise into the parent.

    The import is local on purpose.  ``run_distribution_diagnostics`` is the CLI
    that offers this sweep, so importing it at module scope would close an
    import cycle; deferring it keeps the dependency one-way and costs one
    already-cached import per worker.
    """
    from chapter4_qolu.run_distribution_diagnostics import run_distribution_diagnostics

    try:
        run_distribution_diagnostics(settings)
    except Exception as error:  # noqa: BLE001 - one bad cell must not stop a sweep
        return {"tag": settings.tag, "ok": False, "error": repr(error)}
    return {"tag": settings.tag, "ok": True, "error": None}


def _run_cells(
    cells: Sequence[DiagnosticsSettings], parallel: int
) -> list[dict[str, Any]]:
    """Run every cell, in-process when ``parallel`` is 1, else in a pool."""
    total = len(cells)
    started = time.perf_counter()
    outcomes: list[dict[str, Any]] = []

    def _report(index: int, outcome: dict[str, Any]) -> None:
        logger.log(
            logging.INFO if outcome["ok"] else logging.ERROR,
            "[%d/%d] %s %s (elapsed %s)",
            index,
            total,
            "ok  " if outcome["ok"] else "FAIL",
            outcome["tag"],
            format_duration(time.perf_counter() - started),
        )
        if not outcome["ok"]:
            logger.error("        %s", outcome["error"])

    if max(1, int(parallel)) == 1 or total <= 1:
        for index, settings in enumerate(cells, start=1):
            outcome = _worker(settings)
            outcomes.append(outcome)
            _report(index, outcome)
        return outcomes

    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=min(int(parallel), total)) as pool:
        for index, outcome in enumerate(pool.imap_unordered(_worker, cells), start=1):
            outcomes.append(outcome)
            _report(index, outcome)
    return outcomes


def _settings_record(
    base: DiagnosticsSettings, scenarios: Sequence[int], seeds: Sequence[int]
) -> dict[str, Any]:
    record = base.as_record()
    record.pop("scenario", None)
    record.pop("seed", None)
    record.pop("weights", None)
    record["scenarios"] = [int(value) for value in scenarios]
    record["seeds"] = [int(value) for value in seeds]
    record["configuration"] = base.scale.label
    return record


def sweep_plan(
    base: DiagnosticsSettings,
    cells: Sequence[DiagnosticsSettings],
    parallel: int,
) -> dict[str, Any]:
    """What a sweep would do, with its ETA - the body of ``--dry-run``."""
    paths = sweep_paths(base)
    pending = [cell for cell in cells if cell_record(cell) is None]
    seconds = sum(
        estimate_runtime_seconds([cell.run_spec(diagnostics_paths(cell).summary)])
        for cell in pending
    )
    workers = max(1, min(int(parallel), max(len(pending), 1)))
    return {
        "n_cells": len(cells),
        "n_pending": len(pending),
        "parallel": workers,
        "estimated_serial_seconds": round(seconds, 1),
        "estimated_serial": format_duration(seconds),
        "estimated_wall_clock": format_duration(seconds / workers),
        "cells": [cell.tag for cell in cells],
        "outputs": [str(paths.summary), str(paths.report), str(paths.runs)],
    }


def run_sweep(
    base: DiagnosticsSettings,
    scenarios: Sequence[int] = DEFAULT_SWEEP_SCENARIOS,
    seeds: Sequence[int] = DEFAULT_SWEEP_SEEDS,
    parallel: int = 1,
    dry_run: bool = False,
    execute: bool = True,
) -> dict[str, Any]:
    """Run (or just re-read) every cell and write the two sweep documents."""
    cells = sweep_cells(base, scenarios, seeds)
    paths = sweep_paths(base)
    plan = sweep_plan(base, cells, parallel)
    logger.info(
        "sweep %s: %d cells (%d pending), estimated serial cost %s, "
        "ETA at --parallel %d: %s",
        base.tag,
        plan["n_cells"],
        plan["n_pending"],
        plan["estimated_serial"],
        plan["parallel"],
        plan["estimated_wall_clock"],
    )
    if dry_run:
        return {"tag": base.tag, "dry_run": True, "plan": plan}

    if execute:
        _run_cells(cells, parallel)

    records = [record for record in map(cell_record, cells) if record is not None]
    finished = {record["tag"] for record in records}
    overall = aggregate_records(records, base.headline_target)
    summary: dict[str, Any] = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tag": base.tag,
        "headline_target": base.headline_target,
        "settings": _settings_record(base, scenarios, seeds),
        "plan": plan,
        "n_cells": len(cells),
        "n_complete": len(records),
        "missing": [cell.tag for cell in cells if cell.tag not in finished],
        "cells": records,
        "overall": overall,
        # The headline is one of the three readings; the other two are
        # aggregated beside it rather than dropped, because `n_step_target` is
        # the only one the network is literally regressed onto.
        "by_target": {name: aggregate_records(records, name) for name in TARGET_NAMES},
        "per_scenario": per_scenario_records(records, base.headline_target),
        "per_objective": per_objective_records(records, base.headline_target),
        "verdict": sweep_verdict(records, overall, base.headline_target),
    }
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths.report.write_text(
        render_sweep_report(summary, decimals=base.decimals), encoding="utf-8"
    )
    logger.info("wrote %s", paths.summary)
    logger.info("wrote %s", paths.report)
    logger.info("verdict: %s", summary["verdict"]["statement"])
    return summary
