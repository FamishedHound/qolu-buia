"""Figure (a): before / after land-cover maps, one column per method.

Two rows - the landscape every method in the cell started from (identical by
construction, which the campaign's ``initial_grid_digest`` already asserts) and
the landscape each method left behind - drawn in the LCM aggregate-class
palette with one legend shared by every panel.

The point of the figure is that the *published metrics cannot see* what these
maps show: at the published scale the three metrics are functions of the final
landscape, and when every method develops everything, the maps are the only
place the difference between the policies survives.
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
from chapter4_qolu.figures.data import LoadedCampaign, grids_by_method, runs_for
from chapter4_qolu.figures.palette import (
    CLASS_LEGEND,
    NO_DATA_COLOUR,
    colours_for_raster,
)
from chapter4_qolu.reporting.tables import METHOD_LABELS
from chapter4_qolu.run_record import GRID_NO_DATA

__all__ = ["land_cover_maps"]

_COLUMNS = ("method", "stage", "row", "col", "class_index", "class_label")


def _rows(
    grids: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, (initial, final) in grids.items():
        for stage, values in (("initial", initial), ("final", final)):
            for row_index, row in enumerate(values):
                for col_index, value in enumerate(row):
                    rows.append(
                        {
                            "method": method,
                            "stage": stage,
                            "row": row_index,
                            "col": col_index,
                            "class_index": int(value),
                            "class_label": CLASS_LEGEND.get(int(value), "no data"),
                        }
                    )
    return rows


def _draw(
    grids: Mapping[str, tuple[np.ndarray, np.ndarray]],
    methods: Sequence[str],
    colours: Mapping[int, str],
    title: str,
) -> Any:
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    # Band indices run 0..9 and no-data is -1, so a listed colormap over
    # [-1, 10) with integer boundaries maps every value to its own colour.
    ordered = [NO_DATA_COLOUR] + [colours[index] for index in sorted(colours)]
    cmap = ListedColormap(ordered)
    norm = BoundaryNorm(
        [GRID_NO_DATA - 0.5] + [index + 0.5 for index in range(-1, len(colours))],
        cmap.N,
    )
    figure, axes = new_figure(
        2,
        len(methods),
        figsize=(3.0 * len(methods), 6.4),
        squeeze=False,
        constrained_layout=True,
    )
    for column, method in enumerate(methods):
        initial, final = grids[method]
        for row, (stage, values) in enumerate((("before", initial), ("after", final))):
            axis = axes[row][column]
            axis.imshow(values, cmap=cmap, norm=norm, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(METHOD_LABELS.get(method, method), fontsize=9)
            if column == 0:
                axis.set_ylabel(stage, fontsize=9)
    handles = [Patch(facecolor=NO_DATA_COLOUR, label="Sea / no data")] + [
        Patch(facecolor=colours[index], label=CLASS_LEGEND.get(index, str(index)))
        for index in sorted(colours)
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.09),
    )
    figure.suptitle(title, fontsize=10)
    return figure


def land_cover_maps(
    campaign: LoadedCampaign,
    out_dir: Path,
    scenario: int,
    seed: int,
) -> FigureResult:
    """Before / after land-cover maps for one (scenario, seed) cell."""
    name = f"land_cover_scenario{int(scenario)}_seed{int(seed)}"
    records = runs_for(campaign.records, scenario=scenario, seed=seed)
    grids = grids_by_method(records)
    methods = [
        str(record.get("method", ""))
        for record in records
        if str(record.get("method", "")) in grids
    ]
    paths = [write_rows(out_dir, name, _COLUMNS, _rows(grids))]
    notes: list[str] = []
    if not grids:
        notes.append(
            "no land-cover grids in these records - they predate "
            "run_record.encode_class_grid, so only the CSV header was written"
        )
    elif backend.matplotlib_available():
        figure = _draw(
            grids,
            methods,
            colours_for_raster(campaign.raster_path),
            f"Scenario {scenario}, seed {seed} - land cover before and after",
        )
        paths.extend(save_figure(figure, out_dir, name))
    notes.append(
        "drawn in the 10 LCM2015 aggregate classes; the paper's own legend "
        "uses 8 classes derived from the 21 LCM target classes, a product not "
        "available here"
    )
    return FigureResult(
        name=name,
        paths=tuple(paths),
        caption=(
            f"Land cover before and after development, Scenario {scenario}, "
            f"seed {seed}, one column per method.  Every method in the cell "
            "starts from the same landscape."
        ),
        source=str(campaign.directory),
        command=(
            f"python -m chapter4_qolu.make_figures --campaign {campaign.directory} "
            f"--scenario {scenario} --seed {seed}"
        ),
        notes=tuple(notes),
    )
