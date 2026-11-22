"""Write the Chapter 4 thesis tables from a finished campaign and diagnostics sweep.

Usage::

    python -m chapter4_qolu.make_thesis_tables \\
        --campaign chapter4_qolu/results/campaign_final \\
        --diagnostics chapter4_qolu/results/final_diag/distribution_diagnostics_final

Reads ``<campaign>/summary.json`` and
``<diagnostics>/distribution_diagnostics_sweep.json`` and writes, by default
into ``<campaign>/thesis/``:

* ``table_scenario<k>.tex`` - one booktabs table per scenario;
* ``table_distribution.tex`` - the return-distribution accuracy table;
* a ``.txt`` note beside each, stating what every number is.

Nothing is run and nothing is re-tested: the tables are views of the two JSON
files, so rebuild those first (``run_scenarios --aggregate``) after changing
the reporting code.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from chapter4_qolu.reporting.thesis_tables import (
    THESIS_VIEW,
    render_distribution_note,
    render_distribution_table,
    render_scenario_note,
    render_scenario_table,
)

__all__ = ["main", "write_thesis_tables"]

logger = logging.getLogger(__name__)

#: Stem of the diagnostics sweep document the distribution-accuracy table
#: (``table_distribution.tex``) is read from.
SWEEP_FILE = "distribution_diagnostics_sweep.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _learner(sweep: dict[str, Any]) -> tuple[int, list[int]] | None:
    """``(t_n, syncs)`` from the cells' own summaries, when they all agree.

    Each cell of a sweep points at its ``distribution_diagnostics.json``, whose
    ``learner`` block states the ``t_n`` the stakeholders were built with and
    how many syncs fired.  ``None`` when a summary cannot be read or the cells
    disagree, so the note never states a value it could not check.
    """
    intervals: set[int] = set()
    syncs: list[int] = []
    for cell in sweep.get("cells") or []:
        path = Path(str(cell.get("summary_path", "")))
        if not path.is_file():
            return None
        learner = json.loads(path.read_text(encoding="utf-8")).get("learner") or {}
        if "target_sync_interval" not in learner:
            return None
        intervals.add(int(learner["target_sync_interval"]))
        syncs.extend(int(value) for value in learner.get("target_syncs") or [])
    if len(intervals) != 1:
        return None
    return intervals.pop(), syncs


def write_thesis_tables(
    campaign: Path,
    diagnostics: Path | None = None,
    out: Path | None = None,
    view: str = THESIS_VIEW,
) -> list[Path]:
    """Write every table and note; returns the paths written."""
    campaign = Path(campaign)
    summary_path = campaign / "summary.json"
    summary = _read_json(summary_path)
    target = Path(out) if out is not None else campaign / "thesis"
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    configurations = list(summary.get("configurations", []))
    for configuration in configurations:
        # One configuration (the thesis case) keeps the plain file names.
        suffix = "" if len(configurations) == 1 else f"_{configuration['label']}"
        for cell in configuration.get("scenarios", []):
            stem = f"table_scenario{int(cell['scenario'])}{suffix}"
            tex = target / f"{stem}.tex"
            note = target / f"{stem}.txt"
            tex.write_text(
                render_scenario_table(summary, cell, view=view), encoding="utf-8"
            )
            note.write_text(
                render_scenario_note(
                    summary,
                    cell,
                    view=view,
                    source=summary_path.as_posix(),
                    configuration=configuration,
                ),
                encoding="utf-8",
            )
            written.extend([tex, note])
    if diagnostics is not None:
        sweep_path = Path(diagnostics) / SWEEP_FILE
        sweep = _read_json(sweep_path)
        tex = target / "table_distribution.tex"
        note = target / "table_distribution.txt"
        tex.write_text(render_distribution_table(sweep), encoding="utf-8")
        note.write_text(
            render_distribution_note(
                sweep, source=sweep_path.as_posix(), learner=_learner(sweep)
            ),
            encoding="utf-8",
        )
        written.extend([tex, note])
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, help="campaign results folder")
    parser.add_argument(
        "--diagnostics",
        default=None,
        help="diagnostics sweep folder holding " + SWEEP_FILE,
    )
    parser.add_argument("--out", default=None, help="default: <campaign>/thesis")
    parser.add_argument(
        "--view",
        default=THESIS_VIEW,
        choices=("absolute", "raw"),
        help="metric view of the scenario tables (default: absolute)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    written = write_thesis_tables(
        Path(arguments.campaign),
        Path(arguments.diagnostics) if arguments.diagnostics else None,
        Path(arguments.out) if arguments.out else None,
        view=arguments.view,
    )
    for path in written:
        logger.info("wrote %s", path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
