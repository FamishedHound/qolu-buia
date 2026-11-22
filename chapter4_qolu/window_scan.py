"""The candidate 50 x 50 km windows of the Southeast-England window scan.

The paper says the study region is "the Southeast UK region, 2015" and that the
grid is 50 x 50 parcels of 1 km^2, but it never states an extent.  The scan
therefore searches a grid of candidate windows rather than guessing one, and
reports the whole distribution alongside the window it selects.

This module is the geometry half of that scan - which windows exist and what
land cover each of them holds.  It loads no environment and imports no torch,
so the grid can be inspected instantly.  The scoring lives in
:mod:`chapter4_qolu.scan_scoring` and the driver in
:mod:`chapter4_qolu.run_window_scan`.

Conventions
-----------
A window is named by its **south-west** corner - ``E495_N140`` is the 50 km
square spanning E 495-545 km and N 140-190 km - because that is how an Ordnance
Survey grid square is normally quoted.  The loader wants the **north-west**
corner instead (a raster's first row is its northernmost), which
:attr:`ScanWindow.bng_top_left_km` supplies.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from chapter4_qolu.data.lcm_loader import LCMRaster, bng_origin_to_index
from chapter4_qolu.land_cover import CLASS_LABELS, AggregateClass

__all__ = [
    "MAX_SEA_FRACTION",
    "SOUTH_EAST_ENGLAND",
    "SOUTH_EAST_REGION",
    "SOUTH_EAST_REGION_LABEL",
    "WINDOW_SIZE_KM",
    "WINDOW_STRIDE_KM",
    "BoundingBoxKm",
    "ScanWindow",
    "WindowComposition",
    "candidate_windows",
    "class_share_text",
    "parse_window_label",
    "region_window",
    "retain_windows",
    "window_composition",
    "window_origins",
]

#: A window label as :attr:`ScanWindow.label` writes it, e.g. ``E495_N140`` or
#: ``E410_N60_240x210`` for a rectangle.
_LABEL = re.compile(r"\AE(\d+)_N(\d+)(?:_(\d+)x(\d+))?\Z", re.IGNORECASE)

#: Design note (study-region extent): the paper's Figure 1 shows a South East
#: England study region but gives no coordinates, so the scan searches the
#: British National Grid box E 420-640 km, N 90-250 km - 220 km east-west by
#: 160 km north-south, anchored on Greater London (roughly E 505-555,
#: N 155-200) and reaching the Kent coast in the east (Dover is near E 632,
#: N 142).  It approximates the South East England region; it deliberately
#: neither excludes Greater London and Essex nor reaches the Isle of Wight
#: (south of N 90), and it is *not* a reconstruction of the paper's extent,
#: which the paper does not give.  The box is recorded in
#: ``results/window_scan_*/scan.md`` so a reader can see what was searched.


@dataclass(frozen=True)
class BoundingBoxKm:
    """A British-National-Grid box, in kilometres."""

    east_min: int
    east_max: int
    north_min: int
    north_max: int

    def __post_init__(self) -> None:
        if self.east_max <= self.east_min or self.north_max <= self.north_min:
            raise ValueError(f"empty bounding box: {self}")

    @property
    def description(self) -> str:
        return (
            f"BNG E {self.east_min}-{self.east_max} km, "
            f"N {self.north_min}-{self.north_max} km"
        )

    def as_record(self) -> dict[str, int]:
        return {
            "east_min": self.east_min,
            "east_max": self.east_max,
            "north_min": self.north_min,
            "north_max": self.north_max,
        }


SOUTH_EAST_ENGLAND = BoundingBoxKm(
    east_min=420, east_max=640, north_min=90, north_max=250
)
#: The published extent: 50 x 50 parcels of 1 km^2.
WINDOW_SIZE_KM: int = 50
#: Design note: a 25 km stride, i.e. windows overlapping by half.  Finer would
#: multiply an already multi-hour scan; coarser would let a distinctive
#: landscape fall between two windows.
WINDOW_STRIDE_KM: int = 25
#: Design note: a window with more than 30% sea / out-of-coverage cells is
#: dropped.  Those cells are excluded from the land mask, so they are neither
#: convertible nor developable: past that share the run is no longer a 50 x 50
#: experiment and its metrics are not comparable with the others'.
MAX_SEA_FRACTION: float = 0.30


@dataclass(frozen=True)
class ScanWindow:
    """One candidate window, named by its south-west corner.

    Square by default (the 50 x 50 km windows of the scan).  ``height_km``
    makes it a rectangle, which is what the whole-region reading of the paper's
    Figure 1 needs, and ``name`` overrides the generated label so a named
    region can keep a readable directory name.
    """

    easting_km: int
    northing_km: int
    #: East-west extent, i.e. the number of columns at 1 km parcels.
    size_km: int = WINDOW_SIZE_KM
    #: North-south extent; ``None`` means "square", i.e. ``size_km``.
    height_km: int | None = None
    #: Optional explicit label, e.g. ``SE_region``.
    name: str | None = None

    def __post_init__(self) -> None:
        if self.size_km < 1:
            raise ValueError("size_km must be positive")
        if self.height_km is not None and self.height_km < 1:
            raise ValueError("height_km must be positive")

    @property
    def width_km(self) -> int:
        return int(self.size_km)

    @property
    def rows(self) -> int:
        """North-south extent in parcels (``height_km`` at 1 km parcels)."""
        return int(self.size_km if self.height_km is None else self.height_km)

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, columns)`` of the parcel grid this window cuts."""
        return self.rows, self.width_km

    @property
    def label(self) -> str:
        """Short, filesystem-safe name, e.g. ``E495_N140`` or ``SE_region``."""
        if self.name:
            return self.name
        base = f"E{self.easting_km}_N{self.northing_km}"
        if self.height_km is None or self.height_km == self.size_km:
            return base
        return f"{base}_{self.size_km}x{self.height_km}"

    @property
    def description(self) -> str:
        return (
            f"E {self.easting_km}-{self.easting_km + self.width_km} km, "
            f"N {self.northing_km}-{self.northing_km + self.rows} km"
        )

    @property
    def bng_top_left_km(self) -> tuple[int, int]:
        """North-west corner, which is what the raster loader is pinned to."""
        return self.easting_km, self.northing_km + self.rows

    def as_record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "easting_km": self.easting_km,
            "northing_km": self.northing_km,
            "size_km": self.size_km,
            "width_km": self.width_km,
            "height_km": self.rows,
            "bng_top_left_km": list(self.bng_top_left_km),
            "description": self.description,
        }


#: Design note (whole-region extent): the paper's Figure 1 plots the *entire*
#: South East England region at ~1 km, while the text says "a 50 x 50 grid of
#: parcels, as shown in Figure 1" - the two are inconsistent and only one of
#: them can be run at a time.  The box below, BNG E 410-650 km / N 60-270 km
#: (240 km east-west by 210 km north-south, 50,400 cells at 1 km), is a
#: *bounding box* around that region, not its administrative boundary: it
#: reaches the Isle of Wight in the south-west, the Kent coast in the east and
#: the Oxfordshire / Buckinghamshire edge in the north-west.  It is not a
#: reconstruction of the paper's extent, which is still unknown.
SOUTH_EAST_REGION = BoundingBoxKm(
    east_min=410, east_max=650, north_min=60, north_max=270
)
#: Directory-safe name of the whole-region window.
SOUTH_EAST_REGION_LABEL = "SE_region"


def region_window(box: BoundingBoxKm = SOUTH_EAST_REGION) -> ScanWindow:
    """The whole-region window of ``box``, named :data:`SOUTH_EAST_REGION_LABEL`."""
    return ScanWindow(
        easting_km=box.east_min,
        northing_km=box.north_min,
        size_km=box.east_max - box.east_min,
        height_km=box.north_max - box.north_min,
        name=SOUTH_EAST_REGION_LABEL if box == SOUTH_EAST_REGION else None,
    )


def parse_window_label(text: str, size_km: int = WINDOW_SIZE_KM) -> ScanWindow:
    """Turn a scan label such as ``E495_N140`` back into a :class:`ScanWindow`.

    The inverse of :attr:`ScanWindow.label`, so a window found by the scan can
    be pinned on the command line of an ordinary campaign
    (``run_scenarios --window E495_N140``) or of the sensitivity sweep without
    anyone having to convert corners by hand.  The label names the **south-west**
    corner; :attr:`ScanWindow.bng_top_left_km` supplies the north-west one the
    loader wants.  ``SE_region`` is accepted as the name of the whole-region
    box, and ``E410_N60_240x210`` as its explicit form.
    """
    cleaned = str(text).strip()
    if cleaned.lower() == SOUTH_EAST_REGION_LABEL.lower():
        return region_window()
    match = _LABEL.match(cleaned)
    if not match:
        raise ValueError(
            f"window label {text!r} must look like 'E495_N140', "
            f"'E410_N60_240x210' or '{SOUTH_EAST_REGION_LABEL}' - the "
            "south-west corner of the window in British National Grid kilometres"
        )
    width = int(match.group(3)) if match.group(3) else int(size_km)
    height = int(match.group(4)) if match.group(4) else None
    return ScanWindow(
        easting_km=int(match.group(1)),
        northing_km=int(match.group(2)),
        size_km=width,
        height_km=height,
    )


def window_origins(
    minimum: int, maximum: int, size: int, stride: int
) -> tuple[int, ...]:
    """Window origins tiling ``[minimum, maximum]`` at ``stride``.

    Design note (edge handling): the stride grid runs while a whole window still
    fits inside the box, and a final origin *flush with the far edge* is added
    only when the grid would otherwise leave more than half a stride of the box
    uncovered.  A smaller remainder is not worth a near-duplicate window.  On
    the Southeast box this gives eastings 420, 445, ..., 570, 590 (the last
    flush with E 640) and northings 90, 115, ..., 190, i.e. 8 x 5 = 40 windows.
    """
    if stride < 1:
        raise ValueError("stride must be positive")
    if size < 1:
        raise ValueError("size must be positive")
    if maximum - minimum < size:
        raise ValueError(
            f"box [{minimum}, {maximum}] is narrower than a {size}-unit window"
        )
    origins = list(range(minimum, maximum - size + 1, stride))
    remainder = maximum - (origins[-1] + size)
    if remainder * 2 > stride:
        origins.append(maximum - size)
    return tuple(origins)


def candidate_windows(
    box: BoundingBoxKm = SOUTH_EAST_ENGLAND,
    size_km: int = WINDOW_SIZE_KM,
    stride_km: int = WINDOW_STRIDE_KM,
) -> tuple[ScanWindow, ...]:
    """Every candidate window of the scan, north-west first (reading order)."""
    eastings = window_origins(box.east_min, box.east_max, size_km, stride_km)
    northings = window_origins(box.north_min, box.north_max, size_km, stride_km)
    return tuple(
        ScanWindow(easting_km=easting, northing_km=northing, size_km=size_km)
        for northing in reversed(northings)
        for easting in eastings
    )


@dataclass(frozen=True)
class WindowComposition:
    """What land cover one candidate window actually holds."""

    window: ScanWindow
    n_cells: int
    sea_fraction: float
    #: Share of *all* cells per aggregate-class code, sea excluded and only
    #: for the classes that occur.
    class_shares: Mapping[int, float]

    @property
    def land_fraction(self) -> float:
        return 1.0 - self.sea_fraction

    def as_record(self) -> dict[str, Any]:
        return {
            "window": self.window.as_record(),
            "n_cells": self.n_cells,
            "sea_fraction": self.sea_fraction,
            "class_shares": {
                str(code): share for code, share in self.class_shares.items()
            },
        }


def window_composition(raster: LCMRaster, window: ScanWindow) -> WindowComposition:
    """Cover shares of one window, read straight off the raster."""
    row, col = bng_origin_to_index(raster, *window.bng_top_left_km)
    rows, cols = window.shape
    if row < 0 or col < 0 or row + rows > raster.height or col + cols > raster.width:
        raise ValueError(
            f"window {window.label} ({window.description}) leaves the raster extent "
            f"{raster.height}x{raster.width}"
        )
    patch = raster.fractions[:, row : row + rows, col : col + cols]
    n_cells = rows * cols
    per_class = patch.sum(axis=(1, 2)) / float(n_cells)
    land = float(per_class.sum())
    shares = {
        index + 1: float(value) for index, value in enumerate(per_class) if value > 0.0
    }
    return WindowComposition(
        window=window,
        n_cells=n_cells,
        sea_fraction=float(np.clip(1.0 - land, 0.0, 1.0)),
        class_shares=shares,
    )


def retain_windows(
    raster: LCMRaster,
    windows: Sequence[ScanWindow],
    max_sea_fraction: float = MAX_SEA_FRACTION,
) -> tuple[tuple[WindowComposition, ...], tuple[WindowComposition, ...]]:
    """Split candidate windows into "scan these" and "too much sea"."""
    compositions = [window_composition(raster, window) for window in windows]
    kept = tuple(
        item for item in compositions if item.sea_fraction <= max_sea_fraction + 1e-9
    )
    dropped = tuple(
        item for item in compositions if item.sea_fraction > max_sea_fraction + 1e-9
    )
    return kept, dropped


def class_share_text(shares: Mapping[Any, float], top: int = 3) -> str:
    """The ``top`` largest class shares as ``"improved grassland 42%"`` text."""
    ordered = sorted(
        ((int(code), float(share)) for code, share in shares.items()),
        key=lambda item: (-item[1], item[0]),
    )[:top]
    parts = []
    for code, share in ordered:
        label = CLASS_LABELS.get(AggregateClass(code), f"class {code}")
        parts.append(f"{label.lower()} {share * 100.0:.0f}%")
    return ", ".join(parts)
