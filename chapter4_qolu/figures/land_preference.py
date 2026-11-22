"""The published land-preference pies: what neighbours each land class.

The chapter's land-preference figure is a pie per land class showing how the
**eight-neighbourhood** of that class is composed - "what does arable land sit
next to?" - drawn in the eight-class palette of
``thesis/overleaf/imgs/edit/colours_legend_pie.pdf``, with the largest slice
labelled by its percentage.

The same table is computed for the **initial** grid and for the **final** grid
of a run, so a before / after comparison is one column of the CSV away; the
figures are drawn from the initial grid by default, because that is what the
published figure describes (the landscape, not a policy's effect on it).

Design note (what counts as a neighbour): the eight Moore neighbours inside the
grid.  Cells with no land cover - sea and out-of-coverage - are neither counted
as a centre nor as a neighbour, which is the same exclusion
:meth:`ParcelGrid.land_mask` makes everywhere else.  Saltwater is excluded for
the same reason.
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
from chapter4_qolu.figures.paper_classes import (
    NOT_SEPARABLE_NOTE,
    PAPER_CLASSES,
    aggregate_to_paper,
    paper_class_colours,
    paper_class_names,
)

__all__ = [
    "LEGEND_NAME",
    "land_preference_figures",
    "land_preference_table",
    "paper_class_grid",
]

LEGEND_NAME = "colours_legend_pie"
_COLUMNS = ("stage", "centre_class", "neighbour_class", "count", "share")
#: Design note: a slice smaller than this is left unlabelled, as the published
#: figure labels only the dominant one.
_LABEL_THRESHOLD = 0.0


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def paper_class_grid(values: np.ndarray) -> np.ndarray:
    """Map a band-index grid onto paper-class indices (``-1`` = excluded)."""
    array = np.asarray(values, dtype=np.int64)
    lookup = aggregate_to_paper()
    names = paper_class_names()
    position = {name: index for index, name in enumerate(names)}
    mapped = np.full(array.shape, -1, dtype=np.int64)
    for band, name in lookup.items():
        mapped[array == band] = position[name]
    return mapped


def _neighbour_counts(mapped: np.ndarray, n_classes: int) -> np.ndarray:
    """``counts[centre, neighbour]`` over the eight Moore neighbours."""
    counts = np.zeros((n_classes, n_classes), dtype=np.int64)
    height, width = mapped.shape
    for row_shift in (-1, 0, 1):
        for col_shift in (-1, 0, 1):
            if row_shift == 0 and col_shift == 0:
                continue
            centre = mapped[
                max(0, -row_shift) : height - max(0, row_shift),
                max(0, -col_shift) : width - max(0, col_shift),
            ]
            neighbour = mapped[
                max(0, row_shift) : height - max(0, -row_shift),
                max(0, col_shift) : width - max(0, -col_shift),
            ]
            valid = (centre >= 0) & (neighbour >= 0)
            if not valid.any():
                continue
            np.add.at(
                counts,
                (centre[valid].ravel(), neighbour[valid].ravel()),
                1,
            )
    return counts


def land_preference_table(
    grids: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Neighbour composition per centre class, one block per named stage."""
    names = paper_class_names()
    rows: list[dict[str, Any]] = []
    for stage, values in grids.items():
        counts = _neighbour_counts(paper_class_grid(values), len(names))
        totals = counts.sum(axis=1)
        for centre, name in enumerate(names):
            total = int(totals[centre])
            for neighbour, other in enumerate(names):
                rows.append(
                    {
                        "stage": stage,
                        "centre_class": name,
                        "neighbour_class": other,
                        "count": int(counts[centre, neighbour]),
                        "share": (
                            repr(float(counts[centre, neighbour]) / total)
                            if total
                            else "0.0"
                        ),
                    }
                )
    return rows


def _shares(rows: Sequence[Mapping[str, Any]], stage: str, centre: str) -> list[float]:
    names = paper_class_names()
    lookup = {
        str(row["neighbour_class"]): float(row["share"])
        for row in rows
        if row["stage"] == stage and row["centre_class"] == centre
    }
    return [lookup.get(name, 0.0) for name in names]


def _draw_pie(shares: Sequence[float], title: str) -> Any:
    colours = paper_class_colours()
    names = paper_class_names()
    figure, axes = new_figure(figsize=(3.2, 3.2), constrained_layout=True)
    values = [max(float(share), 0.0) for share in shares]
    if sum(values) <= 0.0:
        axes.text(0.5, 0.5, "no cells", ha="center", va="center", fontsize=9)
        axes.axis("off")
        axes.set_title(title, fontsize=10)
        return figure
    largest = int(np.argmax(values))
    axes.pie(
        values,
        colors=[colours[name] for name in names],
        startangle=90,
        counterclock=False,
        autopct=lambda pct: (
            f"{pct:.0f}%" if pct >= max(values) * 100.0 - 1e-9 else ""
        ),
        textprops={"fontsize": 8, "color": "white"},
        wedgeprops={"linewidth": 0.0},
    )
    del largest  # autopct already singles the dominant slice out
    axes.set_title(title, fontsize=10)
    axes.set_aspect("equal")
    return figure


def _draw_legend() -> Any:
    from matplotlib.patches import Patch

    colours = paper_class_colours()
    names = paper_class_names()
    figure, axes = new_figure(figsize=(4.0, 2.2), constrained_layout=True)
    axes.pie(
        [1.0] * len(names),
        colors=[colours[name] for name in names],
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 0.0},
    )
    axes.set_aspect("equal")
    figure.legend(
        handles=[Patch(facecolor=colours[name], label=name) for name in names],
        loc="center right",
        fontsize=9,
        frameon=True,
    )
    return figure


def land_preference_figures(
    grids: Mapping[str, np.ndarray],
    out_dir: Path,
    stage: str = "initial",
    source: str = "",
    command: str = "",
) -> list[FigureResult]:
    """One pie per land class, plus the shared colour legend.

    ``grids`` maps a stage name (``"initial"``, ``"final"``) to a band-index
    grid; the table covers every stage given, the pies are drawn for ``stage``.
    """
    rows = land_preference_table(grids)
    table_path = write_rows(out_dir, "land_preference", _COLUMNS, rows)
    results: list[FigureResult] = []
    for item in PAPER_CLASSES:
        name = f"land_preference_{_slug(item.name)}"
        paths = [table_path]
        notes: list[str] = []
        if not item.separable:
            notes.append(NOT_SEPARABLE_NOTE)
        if backend.matplotlib_available():
            figure = _draw_pie(_shares(rows, stage, item.name), item.name)
            paths.extend(save_figure(figure, out_dir, name))
        results.append(
            FigureResult(
                name=name,
                paths=tuple(paths),
                caption=(
                    f"Land preference of **{item.name}**: how the eight-cell "
                    f"neighbourhood of a {item.name.lower()} parcel is "
                    f"composed, {stage} landscape.  The dominant slice is "
                    "labelled with its percentage."
                ),
                source=source,
                command=command,
                notes=tuple(
                    notes
                    + [
                        (
                            "shares are over land cells only; sea, "
                            "out-of-coverage and saltwater cells are neither "
                            "centres nor neighbours"
                        ),
                        "the numbers for every stage are in `land_preference.csv`",
                    ]
                ),
            )
        )
    legend_paths = [table_path]
    if backend.matplotlib_available():
        legend_paths.extend(save_figure(_draw_legend(), out_dir, LEGEND_NAME))
    results.append(
        FigureResult(
            name=LEGEND_NAME,
            paths=tuple(legend_paths),
            caption=(
                "The eight land classes of the published legend, in the "
                "published colours."
            ),
            source=source,
            command=command,
            notes=(NOT_SEPARABLE_NOTE,),
        )
    )
    return results
