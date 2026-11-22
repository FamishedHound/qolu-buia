"""The Figure-1 equivalent: the study region before and after 1000 iterations.

The paper's Figure 1 (``thesis/overleaf/imgs/eq1.PNG``) shows the study region
twice, before and after, as a land-cover map with the sea left white.  This
module draws the same pair from a finished run - the initial grid on the left,
the final grid on the right - plus a third variant that greys the landscape out
and shows **only** the parcels the run converted, which is the picture the
published metrics cannot show.

Design note (orientation): the published figure is plotted **transposed** -
west at the top, north on the left - which is what
``imshow(values.T, origin="upper")`` produces from this package's
north-to-south, west-to-east arrays.  The match is visual, against the outline
in ``eq1.PNG``; the paper states no orientation and no coordinates, so
``--map-orientation raster`` draws the ordinary north-up view instead and the
figure caption always says which one it is.

Design note (palette): the eight published land classes of
:mod:`chapter4_qolu.figures.paper_classes`, with sea and out-of-coverage left
**white** as the published figure leaves them.  The published figure's own
colours are not the legend's - it predates the eight-class legend - so the
legend palette is used and the difference is recorded rather than guessed at.
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
from chapter4_qolu.figures.land_preference import paper_class_grid
from chapter4_qolu.figures.paper_classes import (
    NOT_SEPARABLE_NOTE,
    paper_class_colours,
    paper_class_names,
)

__all__ = ["ORIENTATIONS", "region_before_after", "region_converted"]

#: ``paper`` = transposed (west at the top); ``raster`` = north up.
ORIENTATIONS: tuple[str, ...] = ("paper", "raster")
#: Sea / out-of-coverage, as the published figure leaves it.
_NO_DATA_COLOUR = "#ffffff"
#: Colour of an unconverted land parcel in the "converted only" variant.
_BACKGROUND_COLOUR = "#e8e8e8"
_COLUMNS = ("stage", "row", "col", "paper_class")
_SUMMARY_COLUMNS = ("stage", "paper_class", "cells", "share")


def _oriented(values: np.ndarray, orientation: str) -> np.ndarray:
    if orientation not in ORIENTATIONS:
        raise ValueError(
            f"unknown orientation {orientation!r}; expected one of {ORIENTATIONS}"
        )
    return values.T if orientation == "paper" else values


def _class_summary(grids: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    names = paper_class_names()
    rows: list[dict[str, Any]] = []
    for stage, values in grids.items():
        mapped = paper_class_grid(values)
        total = int((mapped >= 0).sum())
        for index, name in enumerate(names):
            cells = int((mapped == index).sum())
            rows.append(
                {
                    "stage": stage,
                    "paper_class": name,
                    "cells": cells,
                    "share": repr(cells / total) if total else "0.0",
                }
            )
    return rows


def _colormap() -> Any:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colours = paper_class_colours()
    names = paper_class_names()
    cmap = ListedColormap([_NO_DATA_COLOUR] + [colours[name] for name in names])
    norm = BoundaryNorm(
        [-1.5] + [index + 0.5 for index in range(-1, len(names))], cmap.N
    )
    return cmap, norm


def _legend_handles() -> list[Any]:
    from matplotlib.patches import Patch

    colours = paper_class_colours()
    return [Patch(facecolor="#ffffff", edgecolor="#bbbbbb", label="Sea / no data")] + [
        Patch(facecolor=colours[name], label=name) for name in paper_class_names()
    ]


def _draw_pair(
    panels: Sequence[tuple[str, np.ndarray]], orientation: str, title: str
) -> Any:
    cmap, norm = _colormap()
    figure, axes = new_figure(
        1,
        len(panels),
        figsize=(5.2 * len(panels), 5.4),
        squeeze=False,
        constrained_layout=True,
    )
    for index, (label, values) in enumerate(panels):
        axis = axes[0][index]
        axis.imshow(
            _oriented(paper_class_grid(values), orientation),
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )
        axis.set_title(label, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    figure.suptitle(title, fontsize=11)
    return figure


def region_before_after(
    initial: np.ndarray,
    final: np.ndarray,
    out_dir: Path,
    tag: str,
    orientation: str = "paper",
    source: str = "",
    command: str = "",
) -> FigureResult:
    """The study region before and after one run, in the published layout."""
    name = f"region_before_after_{tag}"
    grids = {"initial": initial, "final": final}
    paths = [write_rows(out_dir, name, _SUMMARY_COLUMNS, _class_summary(grids))]
    if backend.matplotlib_available():
        figure = _draw_pair(
            [("Before", initial), ("After 1000 iterations", final)],
            orientation,
            f"Study region `{tag}` - land cover before and after",
        )
        paths.extend(save_figure(figure, out_dir, name))
    return FigureResult(
        name=name,
        paths=tuple(paths),
        caption=(
            f"Study region `{tag}`: land cover before (left) and after (right) "
            "the run, in the published eight-class palette, sea left white."
        ),
        source=source,
        command=command,
        notes=(
            f"orientation `{orientation}`"
            + (
                " - transposed so west is at the top and north on the left, "
                "matching the published Figure 1"
                if orientation == "paper"
                else " - north up, the raster's own orientation"
            ),
            NOT_SEPARABLE_NOTE,
            "per-class cell counts and shares for both stages are in the CSV",
        ),
    )


def region_converted(
    initial: np.ndarray,
    final: np.ndarray,
    out_dir: Path,
    tag: str,
    orientation: str = "paper",
    source: str = "",
    command: str = "",
) -> FigureResult:
    """Only the parcels the run converted, over a greyed-out landscape."""
    from chapter4_qolu.land_cover import AggregateClass, band_index

    name = f"region_converted_{tag}"
    urban = band_index(AggregateClass.BUILT_UP)
    land = np.asarray(initial) >= 0
    converted = land & (np.asarray(final) == urban) & (np.asarray(initial) != urban)
    rows = [
        {
            "stage": "converted",
            "row": int(row),
            "col": int(col),
            "paper_class": "Urban",
        }
        for row, col in zip(*np.nonzero(converted))
    ]
    paths = [write_rows(out_dir, name, _COLUMNS, rows)]
    if backend.matplotlib_available():
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        colours = paper_class_colours()
        layer = np.where(converted, 2, np.where(land, 1, 0))
        cmap = ListedColormap([_NO_DATA_COLOUR, _BACKGROUND_COLOUR, colours["Urban"]])
        figure, axes = new_figure(figsize=(5.6, 5.6), constrained_layout=True)
        axes.imshow(
            _oriented(layer, orientation),
            cmap=cmap,
            vmin=0,
            vmax=2,
            origin="upper",
            interpolation="nearest",
        )
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title(
            f"Study region `{tag}` - the {int(converted.sum())} parcels this "
            "run converted",
            fontsize=10,
        )
        figure.legend(
            handles=[
                Patch(
                    facecolor=_NO_DATA_COLOUR,
                    edgecolor="#bbbbbb",
                    label="Sea / no data",
                ),
                Patch(facecolor=_BACKGROUND_COLOUR, label="Unconverted land"),
                Patch(facecolor=colours["Urban"], label="Converted to urban"),
            ],
            loc="lower center",
            ncol=3,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.03),
        )
        paths.extend(save_figure(figure, out_dir, name))
    return FigureResult(
        name=name,
        paths=tuple(paths),
        caption=(
            f"Study region `{tag}`: the {len(rows)} parcels this run converted "
            "to urban, over a greyed-out landscape."
        ),
        source=source,
        command=command,
        notes=(
            (
                "this is what the three published metrics cannot show: they "
                "are functions of the final landscape, so two policies that "
                "develop the same number of parcels in different places can "
                "score alike"
            ),
        ),
    )
