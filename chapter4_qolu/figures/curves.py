"""Figure (b): the per-iteration series, one panel each, with seed bands.

Five panels, all five series every run record already carries:

===========================  =========================================
panel                        what it shows
===========================  =========================================
team reward                  ``R = sum_i w_i R_phi_i + R_society``
urban fraction               how fast the landscape is consumed
cumulative conversions       how many parcels each method approved
well-being                   ``WB = H_occupied / H_total``
conversion alignment         ``C_align``
===========================  =========================================

The line is the mean over seeds and the band is the min-to-max envelope.

Design note (band = min-max, not a confidence interval): at three to five seeds
a standard error is barely defined, and the between-seed spread is typically of
the same order as the between-method gap.  A min-max envelope shows that
without implying a distribution; the per-seed numbers are in the CSV beside
the figure.
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
from chapter4_qolu.figures.data import LoadedCampaign, runs_for, series_by_method
from chapter4_qolu.reporting.tables import METHOD_LABELS

__all__ = ["SERIES_PANELS", "iteration_curves"]

#: ``(series key, axis label, cumulative?)`` per panel, in drawing order.
SERIES_PANELS: tuple[tuple[str, str, bool], ...] = (
    ("team_reward", "Team reward", False),
    ("urban_fraction", "Urban fraction", False),
    ("n_conversions", "Cumulative conversions", True),
    ("well_being", "Well-being", False),
    ("conversion_alignment", "Conversion alignment", False),
)
_COLUMNS = ("panel", "method", "iteration", "mean", "minimum", "maximum", "n_seeds")


def _stack(values: Mapping[int, Sequence[float]], cumulative: bool) -> np.ndarray:
    """Seeds x iterations, truncated to the shortest run in the cell."""
    if not values:
        return np.zeros((0, 0))
    length = min(len(series) for series in values.values())
    stacked = np.array(
        [list(values[seed])[:length] for seed in sorted(values)], dtype=float
    )
    return np.cumsum(stacked, axis=1) if cumulative else stacked


def _panel_rows(
    panel: str, by_method: Mapping[str, Mapping[int, Sequence[float]]], cumulative: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, per_seed in by_method.items():
        stacked = _stack(per_seed, cumulative)
        if stacked.size == 0:
            continue
        for iteration in range(stacked.shape[1]):
            column = stacked[:, iteration]
            rows.append(
                {
                    "panel": panel,
                    "method": method,
                    "iteration": iteration,
                    "mean": repr(float(column.mean())),
                    "minimum": repr(float(column.min())),
                    "maximum": repr(float(column.max())),
                    "n_seeds": int(stacked.shape[0]),
                }
            )
    return rows


def _draw(
    panels: Sequence[tuple[str, str, bool, Mapping[str, Mapping[int, Any]]]],
    methods: Sequence[str],
    title: str,
) -> Any:
    figure, axes = new_figure(
        len(panels),
        1,
        figsize=(7.2, 2.1 * len(panels)),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )
    for index, (_, label, cumulative, by_method) in enumerate(panels):
        axis = axes[index][0]
        for method in methods:
            stacked = _stack(by_method.get(method, {}), cumulative)
            if stacked.size == 0:
                continue
            iterations = np.arange(stacked.shape[1])
            mean = stacked.mean(axis=0)
            line = axis.plot(
                iterations, mean, linewidth=1.2, label=METHOD_LABELS.get(method, method)
            )[0]
            if stacked.shape[0] > 1:
                axis.fill_between(
                    iterations,
                    stacked.min(axis=0),
                    stacked.max(axis=0),
                    alpha=0.18,
                    color=line.get_color(),
                    linewidth=0.0,
                )
        axis.set_ylabel(label, fontsize=8)
        axis.tick_params(labelsize=7)
    axes[-1][0].set_xlabel("iteration", fontsize=8)
    axes[0][0].legend(fontsize=7, frameon=False, ncol=len(methods))
    figure.suptitle(title, fontsize=10)
    return figure


def iteration_curves(
    campaign: LoadedCampaign, out_dir: Path, scenario: int
) -> FigureResult:
    """Per-iteration curves for one scenario, averaged over the campaign's seeds."""
    name = f"curves_scenario{int(scenario)}"
    records = runs_for(campaign.records, scenario=scenario)
    panels = [
        (key, label, cumulative, series_by_method(records, key))
        for key, label, cumulative in SERIES_PANELS
    ]
    rows = [
        row
        for key, _, cumulative, by_method in panels
        for row in _panel_rows(key, by_method, cumulative)
    ]
    paths = [write_rows(out_dir, name, _COLUMNS, rows)]
    methods = [
        method
        for method in campaign.methods
        if any(method in by_method for _, _, _, by_method in panels)
    ]
    if rows and backend.matplotlib_available():
        figure = _draw(
            panels,
            methods,
            f"Scenario {scenario} - per-iteration series (line = mean over "
            "seeds, band = min-max)",
        )
        paths.extend(save_figure(figure, out_dir, name))
    return FigureResult(
        name=name,
        paths=tuple(paths),
        caption=(
            f"Per-iteration series for Scenario {scenario}, one panel per "
            "series; the line is the mean over seeds and the band the "
            "min-to-max envelope."
        ),
        source=str(campaign.directory),
        command=(
            f"python -m chapter4_qolu.make_figures --campaign {campaign.directory} "
            f"--scenario {scenario}"
        ),
        notes=(
            (
                "the band is a min-max envelope over seeds, not a confidence "
                "interval; the per-seed values are in the CSV"
            ),
        ),
    )
