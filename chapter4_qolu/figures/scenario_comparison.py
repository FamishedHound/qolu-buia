"""The published scenario-comparison figure, regenerated from a campaign.

``thesis/overleaf/imgs/edit/scenario_comparison_qolu_buia.pdf`` plots exactly
the values of Tables 4-7: a 2 x 2 grid of panels titled "Scenario 1" to
"Scenario 4", the four methods along the x axis in the published row order
(LToS, P-MADDPG, MitM, QOLU+BUIA), three bars per method labelled *Land Use*,
*Woodland Loss* and *Urban*, and a "Score" axis from 0 to 1.  This module draws
the same layout from a campaign's ``summary.json``.

The ``no_regulator`` ablation, which the published figure does not have, is
drawn first when a campaign ran it, as it is the first row of the tables.

Two deliberate differences from the published figure, both additive:

* **error bars.**  The published figure has none; this one shows +- one
  standard error over seeds, because the between-seed spread is typically of
  the same order as the between-method gap.  Hiding it would reproduce the
  figure and lose that information.
* **an accompanying CSV** at three decimals, so the numbers behind every bar
  are quotable without opening the figure.

Design note (which view): the **absolute** view by default.  The published
scores are absolute 0-1 numbers - every published cell lies between 0.60 and
0.90, so no column contains a 0 or a 1 and they cannot be a cross-method
min-max rescaling - and the absolute view is the only one of the three on the
same footing.  ``--view`` selects another, and the caption always says which.

Design note ("Woodland Loss"): the published legend calls the middle column
*Woodland Loss* while the tables head it *Woodland*.  The metric this package
computes is the **share of the initial woodland and arable area still
standing** (``preservation_of_land_types``), i.e. higher is better, which is
consistent with the tables' orientation and with the published claim that
QOLU+BUIA scores highest.  The published legend wording is used on the figure
and this note is attached to it.
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
from chapter4_qolu.methods import NO_REGULATOR_METHOD, canonical_method
from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS
from chapter4_qolu.reporting.tables import METHOD_LABELS, VIEW_TITLES, ordered_methods

__all__ = [
    "PUBLISHED_METRIC_LEGEND",
    "PUBLISHED_X_ORDER",
    "X_ORDER",
    "scenario_comparison",
]

#: The published legend's wording for the three metrics.
PUBLISHED_METRIC_LEGEND: Mapping[str, str] = {
    "land_use": "Land Use",
    "woodland": "Woodland Loss",
    "urban": "Urban",
}
#: The published x-axis order (the tables' row order).
PUBLISHED_X_ORDER: tuple[str, ...] = ("ltos", "pmaddpg", "mitm", "qolu_buia")
#: The x-axis order drawn: the ablation first, then the published order.
X_ORDER: tuple[str, ...] = (NO_REGULATOR_METHOD, *PUBLISHED_X_ORDER)
#: Bar colours, one per metric.  Design note: a colour-blind-safe trio; the
#: published figure's exact bar colours are not recoverable from the PDF.
_METRIC_COLOURS: Mapping[str, str] = {
    "land_use": "#4c72b0",
    "woodland": "#55a868",
    "urban": "#c44e52",
}
_COLUMNS = ("scenario", "view", "method", "metric", "legend", "mean", "se", "sd", "n")


def _blocks(summary: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        int(block["scenario"]): block
        for configuration in summary.get("configurations", [])
        for block in configuration.get("scenarios", [])
    }


def _methods(blocks: Mapping[int, Mapping[str, Any]]) -> tuple[str, ...]:
    """The campaign's methods along the published x axis.

    A campaign may run *variants* (``pmaddpg_reg_only``, ``mitm_learned``),
    which are not literal members of :data:`PUBLISHED_X_ORDER`, so each method
    is placed by the published method it stands in for.  Without that step a
    variant falls through to the end of the axis and the figure no longer has
    the published column order - which is the one thing this figure is for.
    """
    present = {
        method for block in blocks.values() for method in (block.get("methods") or {})
    }
    position = {name: index for index, name in enumerate(X_ORDER)}
    fallback = {name: index for index, name in enumerate(ordered_methods(present))}
    return tuple(
        sorted(
            present,
            key=lambda method: (
                position.get(canonical_method(method), len(position)),
                fallback.get(method, 0),
            ),
        )
    )


def _rows(
    blocks: Mapping[int, Mapping[str, Any]], methods: Sequence[str], view: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in sorted(blocks):
        block = blocks[scenario]
        for method in methods:
            entry_block = ((block.get("methods") or {}).get(method) or {}).get(view, {})
            for metric in PUBLISHED_METRICS:
                entry = entry_block.get(metric)
                if entry is None:
                    continue
                rows.append(
                    {
                        "scenario": scenario,
                        "view": view,
                        "method": method,
                        "metric": metric,
                        "legend": PUBLISHED_METRIC_LEGEND[metric],
                        "mean": f"{float(entry['mean']):.3f}",
                        "se": f"{float(entry.get('se', 0.0)):.3f}",
                        "sd": f"{float(entry.get('sd', 0.0)):.3f}",
                        "n": int(entry.get("n", 0)),
                    }
                )
    return rows


def _draw(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    scenarios: Sequence[int],
    view: str,
) -> Any:
    lookup = {
        (int(row["scenario"]), str(row["method"]), str(row["metric"])): row
        for row in rows
    }
    # The published figure is 2 x 2 because it has four scenarios; a campaign
    # with fewer gets the smallest grid that holds them, so a single-scenario
    # campaign is one panel rather than one panel and three blanks.
    columns = min(2, max(1, len(scenarios)))
    rows_count = max(1, (len(scenarios) + columns - 1) // columns)
    figure, axes = new_figure(
        rows_count,
        columns,
        figsize=(4.6 * columns, 3.6 * rows_count),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    positions = np.arange(len(methods), dtype=float)
    width = 0.8 / max(len(PUBLISHED_METRICS), 1)
    for index, scenario in enumerate(scenarios):
        axis = axes[index // columns][index % columns]
        for offset_index, metric in enumerate(PUBLISHED_METRICS):
            offset = (offset_index - (len(PUBLISHED_METRICS) - 1) / 2.0) * width
            means = []
            errors = []
            for method in methods:
                row = lookup.get((scenario, method, metric))
                means.append(float(row["mean"]) if row else 0.0)
                errors.append(float(row["se"]) if row else 0.0)
            axis.bar(
                positions + offset,
                means,
                width=width,
                yerr=errors,
                capsize=2.5,
                color=_METRIC_COLOURS[metric],
                label=PUBLISHED_METRIC_LEGEND[metric],
            )
        axis.set_title(f"Scenario {scenario}", fontsize=10)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [METHOD_LABELS.get(method, method) for method in methods],
            fontsize=8,
        )
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Score", fontsize=9)
        axis.tick_params(labelsize=8)
    for index in range(len(scenarios), rows_count * columns):
        axes[index // columns][index % columns].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(PUBLISHED_METRICS),
        fontsize=9,
        frameon=False,
    )
    figure.suptitle(
        f"Scenario comparison - {VIEW_TITLES[view][1]} (mean ± SE over seeds)",
        fontsize=11,
    )
    return figure


def scenario_comparison(
    campaign: LoadedCampaign, out_dir: Path, view: str = "absolute"
) -> FigureResult:
    """The published 2 x 2 scenario-comparison layout, from one campaign."""
    if view not in VIEW_TITLES:
        raise ValueError(f"unknown view {view!r}; expected one of {tuple(VIEW_TITLES)}")
    name = f"scenario_comparison_{campaign.tag}"
    blocks = _blocks(campaign.summary)
    methods = _methods(blocks)
    rows = _rows(blocks, methods, view)
    paths = [write_rows(out_dir, name, _COLUMNS, rows)]
    if rows and backend.matplotlib_available():
        paths.extend(
            save_figure(_draw(rows, methods, sorted(blocks), view), out_dir, name)
        )
    return FigureResult(
        name=name,
        paths=tuple(paths),
        caption=(
            "Scenario comparison in the published layout: one panel per "
            "scenario, methods in the published order, three bars per method "
            f"(Land Use, Woodland Loss, Urban), {VIEW_TITLES[view][2]}."
        ),
        source=str(campaign.directory),
        command=(
            f"python -m chapter4_qolu.make_figures --campaign {campaign.directory} "
            f"--view {view}"
        ),
        notes=(
            (
                "the published figure carries no error bars; these are ± "
                "one standard error over seeds"
            ),
            (
                "'Woodland Loss' is the published legend's wording; the "
                "metric is the share of the initial woodland and arable area "
                "still standing, so higher is better, as in the published "
                "tables"
            ),
            (
                f"drawn on view ({view}); the published values are absolute "
                "0-1 scores whose normalisation constants are unknown"
            ),
        ),
    )
