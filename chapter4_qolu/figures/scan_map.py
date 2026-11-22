"""Figure (d): the searched windows, coloured by their stage-1 score.

The Southeast bounding box with every candidate window drawn in place,
shaded by the primary score (how many of the three Scenario-1 metrics put
QOLU + BUIA strictly first), the chosen window outlined, over a land / sea mask
read from the raster when one is at hand.

This is the figure that makes the *selection* visible: a reader can see how
many windows were tried, how they scored and which one was picked, which is
exactly what ``scan.md``'s "how the window was chosen" section says in words.
"""

from __future__ import annotations

import itertools
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

__all__ = ["land_mask_for_box", "scan_score_map"]

_COLUMNS = (
    "window",
    "east_min_km",
    "east_max_km",
    "north_min_km",
    "north_max_km",
    "sea_fraction",
    "first_place_count",
    "kendall_sum",
    "reference_minmax_mean",
    "chosen",
)


def land_mask_for_box(
    raster_path: Path | str | None, box: Mapping[str, Any]
) -> np.ndarray | None:
    """Boolean land mask of the search box, or ``None`` without a raster.

    Rows run north to south, as a raster's do, so the array can be drawn with
    ``origin='upper'`` and an extent in kilometres without transposing.
    """
    if not raster_path or not box:
        return None
    path = Path(raster_path)
    if not path.is_file():
        return None
    from chapter4_qolu.data.lcm_loader import bng_origin_to_index, load_lcm_raster

    raster = load_lcm_raster(path)
    row, col = bng_origin_to_index(
        raster, float(box["east_min"]), float(box["north_max"])
    )
    height = int(box["north_max"]) - int(box["north_min"])
    width = int(box["east_max"]) - int(box["east_min"])
    if row < 0 or col < 0 or row + height > raster.height or col + width > raster.width:
        return None
    patch = raster.fractions[:, row : row + height, col : col + width]
    return np.asarray(patch.sum(axis=0) > 1e-4, dtype=bool)


def _entries(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list((summary.get("stage1") or {}).get("windows") or [])


def _winner_label(summary: Mapping[str, Any]) -> str | None:
    winner = (summary.get("selection") or {}).get("winner") or {}
    return str(winner.get("label")) if winner.get("label") else None


def _rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    winner = _winner_label(summary)
    rows: list[dict[str, Any]] = []
    for entry in _entries(summary):
        window = entry["window"]
        score = entry.get("score", {})
        composition = entry.get("composition", {})
        width = int(window.get("width_km", window.get("size_km", 50)))
        height = int(window.get("height_km", window.get("size_km", 50)))
        rows.append(
            {
                "window": window["label"],
                "east_min_km": int(window["easting_km"]),
                "east_max_km": int(window["easting_km"]) + width,
                "north_min_km": int(window["northing_km"]),
                "north_max_km": int(window["northing_km"]) + height,
                "sea_fraction": repr(float(composition.get("sea_fraction", 0.0))),
                "first_place_count": int(score.get("first_place_count", 0)),
                "kendall_sum": repr(float(score.get("kendall_sum", 0.0))),
                "reference_minmax_mean": repr(
                    float(score.get("reference_minmax_mean", 0.0))
                ),
                "chosen": int(window["label"] == winner),
            }
        )
    return rows


#: The primary score's fixed range, so two scans are comparable at a glance.
_SCORE_MAX = 3.0


def _tile_size(rows: Sequence[Mapping[str, Any]]) -> float:
    """Stride of the window grid, i.e. the largest non-overlapping tile.

    Design note (why tiles, not footprints): the scan's windows overlap by half
    - 50 km squares at a 25 km stride - so drawing each at its true extent
    makes every window hide its neighbours and the map unreadable.  Each is
    therefore drawn as a **stride-sized tile centred on the window's centre**,
    which tiles the box exactly once; the chosen window's *full* 50 km
    footprint is outlined on top, so the real extent is still visible.
    """
    eastings = sorted({float(row["east_min_km"]) for row in rows})
    northings = sorted({float(row["north_min_km"]) for row in rows})
    gaps = [
        second - first
        for values in (eastings, northings)
        for first, second in itertools.pairwise(values)
        if second > first
    ]
    if gaps:
        return float(min(gaps))
    first = rows[0]
    return float(first["east_max_km"]) - float(first["east_min_km"])


def _draw(
    rows: Sequence[Mapping[str, Any]],
    box: Mapping[str, Any],
    mask: np.ndarray | None,
    title: str,
) -> Any:
    import matplotlib
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    figure, axes = new_figure(figsize=(8.0, 6.2), constrained_layout=True)
    extent = (
        float(box["east_min"]),
        float(box["east_max"]),
        float(box["north_min"]),
        float(box["north_max"]),
    )
    if mask is not None:
        # Land pale grey, sea white, so the coastline is legible under the
        # score tiles without competing with them for attention.
        axes.imshow(
            mask.astype(float),
            cmap=matplotlib.colors.ListedColormap(["#ffffff", "#d9d9d9"]),
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )
    norm = Normalize(vmin=0.0, vmax=_SCORE_MAX)
    colours = matplotlib.colormaps["viridis"]
    size = _tile_size(rows)
    chosen_row = next((row for row in rows if int(row["chosen"])), None)
    for row in rows:
        centre_east = (float(row["east_min_km"]) + float(row["east_max_km"])) / 2.0
        centre_north = (float(row["north_min_km"]) + float(row["north_max_km"])) / 2.0
        score = float(row["first_place_count"])
        axes.add_patch(
            Rectangle(
                (centre_east - size / 2.0, centre_north - size / 2.0),
                size,
                size,
                facecolor=colours(norm(score)),
                edgecolor="white",
                linewidth=0.6,
            )
        )
        axes.text(
            centre_east,
            centre_north,
            f"{row['window']}\n{int(score)}/3",
            ha="center",
            va="center",
            fontsize=5.0,
            color="white" if score < _SCORE_MAX * 0.6 else "black",
            linespacing=1.1,
        )
    if chosen_row is not None:
        axes.add_patch(
            Rectangle(
                (float(chosen_row["east_min_km"]), float(chosen_row["north_min_km"])),
                float(chosen_row["east_max_km"]) - float(chosen_row["east_min_km"]),
                float(chosen_row["north_max_km"]) - float(chosen_row["north_min_km"]),
                facecolor="none",
                edgecolor="crimson",
                linewidth=2.6,
                zorder=5,
            )
        )
    axes.set_xlim(extent[0], extent[1])
    axes.set_ylim(extent[2], extent[3])
    axes.set_xlabel("BNG easting (km)", fontsize=8)
    axes.set_ylabel("BNG northing (km)", fontsize=8)
    axes.set_aspect("equal")
    axes.tick_params(labelsize=7)
    bar = figure.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"), ax=axes, shrink=0.75
    )
    bar.set_ticks([0.0, 1.0, 2.0, 3.0])
    bar.set_label("metrics with QOLU+BUIA strictly first (0-3)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    axes.set_title(title, fontsize=10)
    return figure


def scan_score_map(
    summary: Mapping[str, Any],
    out_dir: Path,
    raster_path: Path | str | None = None,
) -> FigureResult:
    """The whole searched grid, coloured by stage-1 score, winner outlined."""
    tag = str(summary.get("tag", "scan"))
    name = f"scan_map_{tag}"
    box = (summary.get("grid") or {}).get("bounding_box_km") or {}
    rows = _rows(summary)
    paths = [write_rows(out_dir, name, _COLUMNS, rows)]
    if rows and box and backend.matplotlib_available():
        figure = _draw(
            rows,
            box,
            land_mask_for_box(raster_path, box),
            f"Window scan `{tag}` - {len(rows)} windows searched, "
            f"winner outlined in red",
        )
        paths.extend(save_figure(figure, out_dir, name))
    return FigureResult(
        name=name,
        paths=tuple(paths),
        caption=(
            f"The {len(rows)} candidate windows of scan `{tag}`, shaded by how "
            "many of the three Scenario-1 metrics put QOLU+BUIA strictly "
            "first; the selected window is outlined."
        ),
        source=str(out_dir),
        command=f"python -m chapter4_qolu.make_figures --scan <scan dir> --out {out_dir}",
        notes=(
            (
                "the shading is the *primary* score only, on a fixed 0-3 "
                "scale; the full lexicographic order is in scan.md and "
                "scan_summary.json"
            ),
            (
                "the scan's windows overlap by half, so each is drawn as a "
                "stride-sized tile at its centre; the chosen window's full "
                "50 km footprint is outlined in red on top"
            ),
        ),
    )
