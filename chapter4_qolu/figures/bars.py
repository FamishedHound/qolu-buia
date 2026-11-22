"""Figure (c): the published-style tables as grouped bars, mean +- SE.

One figure per (scenario, view): three metric groups on the x axis, one bar per
method, error bars at +- one standard error over seeds - the paper's Figure 5
layout, drawn from ``summary.json`` so the bars and the tables can never
disagree.

The view is never implicit.  The same cell looks different in view (a) RAW,
view (b) absolute-normalised and view (c) cross-method min-max, and the caption
of every figure says which it is showing; the published tables are none of the
three (see :mod:`chapter4_qolu.reporting.published`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from chapter4_qolu.figures import backend
from chapter4_qolu.figures.backend import (
    FigureResult,
    new_figure,
    save_figure,
    write_rows,
)
from chapter4_qolu.figures.data import LoadedCampaign
from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS
from chapter4_qolu.reporting.tables import (
    METHOD_LABELS,
    METRIC_LABELS,
    VIEW_TITLES,
    ordered_methods,
)

__all__ = ["VIEWS", "table_bars"]

#: The three readings, in the order ``tables.md`` prints them.
VIEWS: tuple[str, ...] = ("minmax", "raw", "absolute")
_COLUMNS = ("scenario", "view", "method", "metric", "mean", "sd", "se", "n")


def _scenario_blocks(
    summary: Mapping[str, Any], scenario: int
) -> list[Mapping[str, Any]]:
    return [
        block
        for configuration in summary.get("configurations", [])
        for block in configuration.get("scenarios", [])
        if int(block.get("scenario", -1)) == int(scenario)
    ]


def _rows(
    blocks: Sequence[Mapping[str, Any]], scenario: int, view: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in blocks:
        for method, method_block in (block.get("methods") or {}).items():
            for metric in PUBLISHED_METRICS:
                entry = ((method_block or {}).get(view) or {}).get(metric)
                if entry is None:
                    continue
                rows.append(
                    {
                        "scenario": int(scenario),
                        "view": view,
                        "method": method,
                        "metric": metric,
                        "mean": repr(float(entry.get("mean", 0.0))),
                        "sd": repr(float(entry.get("sd", 0.0))),
                        "se": repr(float(entry.get("se", 0.0))),
                        "n": int(entry.get("n", 0)),
                    }
                )
    return rows


def _draw(rows: Sequence[Mapping[str, Any]], scenario: int, view: str) -> Any:
    methods = ordered_methods({str(row["method"]) for row in rows})
    lookup = {(str(row["method"]), str(row["metric"])): row for row in rows}
    positions = np.arange(len(PUBLISHED_METRICS), dtype=float)
    width = 0.8 / max(len(methods), 1)
    figure, axes = new_figure(figsize=(6.4, 3.6), constrained_layout=True)
    for index, method in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2.0) * width
        means = []
        errors = []
        for metric in PUBLISHED_METRICS:
            row = lookup.get((method, metric))
            means.append(float(row["mean"]) if row else 0.0)
            errors.append(float(row["se"]) if row else 0.0)
        axes.bar(
            positions + offset,
            means,
            width=width,
            yerr=errors,
            capsize=2.5,
            label=METHOD_LABELS.get(method, method),
        )
    axes.set_xticks(positions)
    axes.set_xticklabels([METRIC_LABELS[metric] for metric in PUBLISHED_METRICS])
    axes.set_ylabel(VIEW_TITLES[view][1].split(":")[0])
    axes.legend(fontsize=7, frameon=False, ncol=2)
    axes.set_title(
        f"Scenario {scenario} - {VIEW_TITLES[view][1]} (mean ± SE over seeds)",
        fontsize=9,
    )
    axes.tick_params(labelsize=8)
    return figure


def table_bars(
    campaign: LoadedCampaign, out_dir: Path, scenario: int, view: str = "minmax"
) -> FigureResult:
    """Grouped bars of one scenario's published-style table, in one view."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    name = f"table_bars_scenario{int(scenario)}_{view}"
    rows = _rows(_scenario_blocks(campaign.summary, scenario), scenario, view)
    paths = [write_rows(out_dir, name, _COLUMNS, rows)]
    if rows and backend.matplotlib_available():
        paths.extend(save_figure(_draw(rows, scenario, view), out_dir, name))
    return FigureResult(
        name=name,
        paths=tuple(paths),
        caption=(
            f"Scenario {scenario}, {VIEW_TITLES[view][2]}.  Bars are the mean "
            "over seeds and the error bars one standard error."
        ),
        source=str(campaign.directory),
        command=(
            f"python -m chapter4_qolu.make_figures --campaign {campaign.directory} "
            f"--scenario {scenario} --view {view}"
        ),
        notes=(
            (
                "the published Tables 4-7 are on none of the three views - see "
                "reporting/published.py - so these bars are not comparable "
                "with the paper's Figure 5 cell by cell"
            ),
        ),
    )
