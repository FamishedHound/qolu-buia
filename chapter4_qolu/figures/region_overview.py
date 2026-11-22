"""Figure 4.1 of the thesis: the region around the study window, and the window
before and after one run.

The large panel is the land cover of the South East England bounding box
(:data:`chapter4_qolu.window_scan.SOUTH_EAST_REGION`, BNG E 410-650 km,
N 60-270 km), read from the LCM2015 1 km dominant aggregate class raster, with
the study window outlined.  The two small panels are the window's grid at the
start and at the end of one run.  All three are drawn north up, in the
eight-class palette of the published legend (:mod:`.paper_classes`), with sea
and out-of-coverage cells left white.

The window is cut from the same raster as the region, and the script checks
that the cut equals the run's stored initial grid before it draws anything.

Usage (from the folder that contains the package)::

    python -m chapter4_qolu.figures.region_overview \\
        --campaign chapter4_qolu/results/campaign_final \\
        --raster chapter4_qolu/data_raw/lcm2015_gb_1km/lcm2015_gb_1km_dominant_aggregate_class.img \\
        --scenario 1 --seed 0 --out chapter4_qolu/results/campaign_final/figures
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from chapter4_qolu.figures import backend
from chapter4_qolu.figures.backend import FigureResult, save_figure, write_rows
from chapter4_qolu.figures.land_preference import paper_class_grid
from chapter4_qolu.figures.region_maps import (
    _SUMMARY_COLUMNS,
    _class_summary,
    _colormap,
    _legend_handles,
)
from chapter4_qolu.window_scan import SOUTH_EAST_REGION, BoundingBoxKm

__all__ = ["band_grid", "region_grid", "window_offset", "region_with_window", "main"]

#: The campaign's study window is 50 x 50 parcels of 1 km^2.
WINDOW_SIZE_KM = 50
_TICK_STEP_KM = 50


def band_grid(fractions: np.ndarray) -> np.ndarray:
    """Band-index grid (0-9, ``-1`` = sea / no data) of a ``(10, rows, cols)`` cover stack."""
    stack = np.asarray(fractions)
    land = stack.sum(axis=0) > 0
    return np.where(land, stack.argmax(axis=0), -1).astype(np.int64)


def region_grid(raster, box: BoundingBoxKm = SOUTH_EAST_REGION) -> np.ndarray:
    """The band-index grid of ``box``, north up, one row per km of northing."""
    from chapter4_qolu.data.lcm_loader import bng_origin_to_index

    row, col = bng_origin_to_index(raster, box.east_min, box.north_max)
    rows, cols = box.north_max - box.north_min, box.east_max - box.east_min
    if row < 0 or col < 0 or row + rows > raster.height or col + cols > raster.width:
        raise ValueError(f"{box.description} leaves the raster extent")
    return band_grid(raster.fractions[:, row : row + rows, col : col + cols])


def window_offset(
    box: BoundingBoxKm, top_left_km: Sequence[float], size_km: int = WINDOW_SIZE_KM
) -> tuple[int, int]:
    """``(row, col)`` of the window's north-west parcel inside the region grid."""
    east, north = float(top_left_km[0]), float(top_left_km[1])
    row, col = int(round(box.north_max - north)), int(round(east - box.east_min))
    rows, cols = box.north_max - box.north_min, box.east_max - box.east_min
    if row < 0 or col < 0 or row + size_km > rows or col + size_km > cols:
        raise ValueError(f"window at E {east:g}, N {north:g} km is not inside {box.description}")
    return row, col


def _draw(region, box, top_left_km, size_km, initial, final, after_label):
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cmap, norm = _colormap()
    figure = plt.figure(figsize=(8.6, 4.9), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.75, 1.0))
    big = figure.add_subplot(grid[:, 0])
    big.imshow(
        paper_class_grid(region),
        cmap=cmap,
        norm=norm,
        origin="upper",
        interpolation="nearest",
        extent=(box.east_min, box.east_max, box.north_min, box.north_max),
    )
    east, north = float(top_left_km[0]), float(top_left_km[1])
    big.add_patch(
        Rectangle((east, north - size_km), size_km, size_km, fill=False, edgecolor="black", linewidth=1.6)
    )
    big.annotate(
        "study window",
        xy=(east + size_km, north),
        xytext=(east + size_km + 8, north + 14),
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 0.8},
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    first_e = (box.east_min // _TICK_STEP_KM + 1) * _TICK_STEP_KM
    first_n = (box.north_min // _TICK_STEP_KM + 1) * _TICK_STEP_KM
    big.set_xticks(range(first_e, box.east_max + 1, _TICK_STEP_KM))
    big.set_yticks(range(first_n, box.north_max + 1, _TICK_STEP_KM))
    big.tick_params(labelsize=8.5)
    big.set_xlabel("km east (British National Grid)", fontsize=9)
    big.set_ylabel("km north (British National Grid)", fontsize=9)
    big.set_title("South East England, 2015", fontsize=10.5)
    for position, (title, values) in enumerate(
        (("Study window: before", initial), (after_label, final))
    ):
        axis = figure.add_subplot(grid[position, 1])
        axis.imshow(paper_class_grid(values), cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.09),
    )
    return figure


def region_with_window(
    region: np.ndarray,
    initial: np.ndarray,
    final: np.ndarray,
    out_dir: Path,
    tag: str,
    top_left_km: Sequence[float],
    box: BoundingBoxKm = SOUTH_EAST_REGION,
    size_km: int = WINDOW_SIZE_KM,
    after_label: str = "After 1000 iterations",
    source: str = "",
    command: str = "",
) -> FigureResult:
    """The region with the window outlined, and the window before and after one run."""
    row, col = window_offset(box, top_left_km, size_km)
    cut = np.asarray(region)[row : row + size_km, col : col + size_km]
    if cut.shape != np.asarray(initial).shape or not np.array_equal(cut, np.asarray(initial)):
        raise ValueError(
            "the window cut from the region does not equal the run's initial grid: "
            "check the raster and the window corner"
        )
    name = f"region_overview_{tag}"
    grids = {"region": region, "initial": initial, "final": final}
    paths = [write_rows(out_dir, name, _SUMMARY_COLUMNS, _class_summary(grids))]
    if backend.matplotlib_available():
        figure = _draw(region, box, top_left_km, size_km, initial, final, after_label)
        paths.extend(save_figure(figure, out_dir, name))
    return FigureResult(
        name=name,
        paths=tuple(paths),
        caption=(
            f"Land cover of {box.description} with the study window outlined, and the "
            "window before and after one run, north up, in the published eight-class palette."
        ),
        source=source,
        command=command,
        notes=("per-class cell counts and shares of the region and of both stages are in the CSV",),
    )


def main(argv: Sequence[str] | None = None) -> int:
    from chapter4_qolu.data.lcm_loader import load_lcm_raster
    from chapter4_qolu.figures.data import grids_by_method, load_campaign, runs_for

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--raster", type=Path, required=True)
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="qolu_buia")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    campaign = load_campaign(args.campaign)
    top_left = campaign.summary.get("campaign", {}).get("window_bng_top_left_km")
    if not top_left:
        raise SystemExit(f"{args.campaign}/summary.json records no window corner")
    grids = grids_by_method(runs_for(campaign.records, scenario=args.scenario, seed=args.seed))
    if args.method not in grids:
        raise SystemExit(
            f"no land-cover grids for {args.method}, scenario {args.scenario}, seed {args.seed}: "
            "the published records carry none; rerun the campaign to regenerate them"
        )
    initial, final = grids[args.method]
    region = region_grid(load_lcm_raster(args.raster))
    out_dir = args.out or (args.campaign / "figures")
    command = (
        f"python -m chapter4_qolu.figures.region_overview --campaign {args.campaign} "
        f"--raster {args.raster} --scenario {args.scenario} --seed {args.seed}"
    )
    result = region_with_window(
        region, initial, final, out_dir, campaign.tag, top_left,
        source=str(campaign.directory), command=command,
    )
    for path in result.paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
