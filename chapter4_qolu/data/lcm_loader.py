"""Load a UKCEH LCM aggregate-class raster into a parcel grid.

Two products are read, and the band count decides which:

*percentage cover* (ten bands)
    the product the paper uses - one band per aggregate class, each giving the
    percentage of the 1 km cell covered by that class.

*dominant aggregate class* (one band)
    the GB-wide single-band product, whose cells hold the class *code* of the
    aggregate class that covers most of the cell (1-10, with 0 for sea and
    out-of-coverage).  It is expanded into the same ten-band "percentage cover"
    representation, with 100% of the dominant class and 0 elsewhere, so
    everything downstream - :class:`~chapter4_qolu.env.parcel_grid.ParcelGrid`,
    the attractiveness field, the metrics - is untouched.  Code 0 becomes an
    all-zero column, which is exactly how the percentage product's no-data
    cells already behave: ``ParcelGrid.land_mask`` excludes them.

Region caveat
-------------
The paper places its study region in south-east UK.  The *Northern Ireland*
product is a convenient stand-in for smoke runs and tests, and a placeholder
only.  :func:`load_lcm_raster` warns whenever the file name looks like an NI
product so the substitution can never pass unnoticed.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chapter4_qolu.data.raster_backend import Backend, read_raster
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES

__all__ = [
    "DOMINANT_CLASS_NO_DATA",
    "LCMRaster",
    "bng_origin_to_index",
    "is_placeholder_region",
    "load_lcm_raster",
    "resolve_window_origin",
    "select_region",
    "select_region_of_interest",
]

logger = logging.getLogger(__name__)

#: Cell value of the *dominant aggregate class* product meaning "sea or outside
#: the mapped area".  Design note: 0 is no-data.  The product's own colour table
#: (``*.img.clr``) gives 0 a neutral grey and codes 1-10 the ten published
#: aggregate-class colours, and the GB raster's 0 cells are the sea; the class
#: codes themselves are the published aggregate-class numbers, which is the
#: order :mod:`chapter4_qolu.land_cover` already uses.
DOMINANT_CLASS_NO_DATA: int = 0

#: File-name fragments that indicate the Northern-Ireland placeholder product.
_PLACEHOLDER_PATTERNS = (
    re.compile(r"(^|[^a-z])ni([^a-z]|$)"),
    re.compile(r"northern[_\- ]?ireland"),
)

#: The 1 km products are already at parcel resolution; the paper's Figure 1
#: instead describes "blocks of 40 x 40 pixel units", which is exactly one
#: 1 km^2 parcel at the 25 m resolution of the LCM raster.
PARCEL_SIZE_M: float = 1000.0

#: How far a BNG origin may sit from a parcel boundary, in parcels, before it
#: is rejected.  Design note: 1e-6 parcels (a millimetre at 1 km cells) - enough
#: for floating-point noise in a transform, far too little to hide a genuine
#: half-cell offset.
_INDEX_TOLERANCE: float = 1e-6


@dataclass(frozen=True)
class LCMRaster:
    """Land cover fractions read from an LCM percentage-cover product."""

    fractions: np.ndarray  # (n_classes, height, width), values in [0, 1]
    path: Path
    pixel_size_m: float
    crs: str | None = None
    #: Map coordinates ``(easting, northing)`` of the raster's top-left corner,
    #: in CRS units (metres for the British National Grid).  ``None`` when the
    #: source carried no transform, in which case only ``(row, col)`` windows
    #: can be selected.
    origin_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        array = np.asarray(self.fractions, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(f"fractions must be 3-D, got shape {array.shape}")
        if array.shape[0] != N_AGGREGATE_CLASSES:
            raise ValueError(
                f"expected {N_AGGREGATE_CLASSES} aggregate-class bands, got {array.shape[0]}"
            )
        if self.pixel_size_m <= 0.0:
            raise ValueError("pixel_size_m must be positive")
        array.setflags(write=False)
        object.__setattr__(self, "fractions", array)
        object.__setattr__(self, "path", Path(self.path))

    @property
    def height(self) -> int:
        return int(self.fractions.shape[1])

    @property
    def width(self) -> int:
        return int(self.fractions.shape[2])

    def aggregate_pixels(self, block: int) -> "LCMRaster":
        """Average ``block x block`` pixels into one parcel.

        Needed for the 25 m LCM raster (``block=40`` gives 1 km^2 parcels); the
        1 km product used here is already at parcel resolution (``block=1``).
        """
        if block < 1:
            raise ValueError("block must be a positive number of pixels")
        if block == 1:
            return self
        if self.height % block or self.width % block:
            raise ValueError(
                f"raster {self.height}x{self.width} is not divisible into {block}x{block} blocks"
            )
        reshaped = self.fractions.reshape(
            self.fractions.shape[0],
            self.height // block,
            block,
            self.width // block,
            block,
        )
        return LCMRaster(
            fractions=reshaped.mean(axis=(2, 4)).astype(np.float32),
            path=self.path,
            pixel_size_m=self.pixel_size_m * block,
            crs=self.crs,
            # Block averaging shrinks the raster towards its top-left corner,
            # which therefore stays put.
            origin_xy=self.origin_xy,
        )


def is_placeholder_region(path: Path | str) -> bool:
    """Heuristic: does this file look like the Northern-Ireland placeholder?"""
    name = Path(path).name.lower()
    return any(pattern.search(name) for pattern in _PLACEHOLDER_PATTERNS)


def load_lcm_raster(
    path: Path | str,
    backend: Backend = "auto",
    warn_on_placeholder: bool = True,
    expected_pixel_size_m: float | None = PARCEL_SIZE_M,
) -> LCMRaster:
    """Read an LCM aggregate-class raster as per-parcel cover fractions.

    Ten bands are read as the published *percentage cover* product; a single
    band is read as the *dominant aggregate class* product and expanded into
    the same ten-band form.
    """
    path = Path(path)
    raster = read_raster(path, backend=backend)

    n_bands = int(raster.values.shape[0])
    if n_bands == 1:
        fractions = _dominant_classes_to_fractions(raster.values[0], path)
    elif n_bands == N_AGGREGATE_CLASSES:
        fractions = _percentages_to_fractions(raster.values)
    else:
        raise ValueError(
            f"{path.name} has {n_bands} bands; expected {N_AGGREGATE_CLASSES} "
            "(percentage cover) or 1 (dominant aggregate class)"
        )

    if warn_on_placeholder and is_placeholder_region(path):
        message = (
            f"{path.name} looks like Northern-Ireland LCM data: placeholder region "
            "data, not the paper's actual study region."
        )
        logger.warning(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    pixel_size = raster.pixel_size_m
    if pixel_size is None:
        pixel_size = expected_pixel_size_m or PARCEL_SIZE_M
        logger.warning(
            "%s has no pixel-size metadata; assuming %.1f m parcels",
            path.name,
            pixel_size,
        )

    return LCMRaster(
        fractions=fractions,
        path=path,
        pixel_size_m=float(pixel_size),
        crs=raster.crs,
        origin_xy=raster.origin_xy,
    )


def _dominant_classes_to_fractions(values: np.ndarray, path: Path) -> np.ndarray:
    """One-hot the dominant-class codes into the ten cover bands.

    Design note (class order): the cell values are the published LCM2015
    aggregate-class numbers, so code ``k`` is band ``k - 1`` of
    :mod:`chapter4_qolu.land_cover` - broadleaf, coniferous, arable, improved
    grassland, semi-natural grassland, mountain/heath/bog, saltwater,
    freshwater, coastal, built-up.  Checked twice before use: against the
    product's own ``*.img.clr`` colour table (1 red, 2 dark green, 3 brown,
    4 bright green, 5 olive, 6 purple, 7 navy, 8 blue, 9 pale yellow, 10 grey -
    the published aggregate-class palette) and against the class shares of a
    known 50 x 50 km window over Surrey / West Kent, where the codes come out
    as 42% improved grassland, 37% built-up, 11% broadleaf.

    Design note (what a dominant class means for cover): a parcel is recorded as
    100% of its dominant class.  The published product is a percentage mix; the
    dominant-class product does not carry the mix, so the only faithful reading
    is "all of the majority class".  Everything downstream consumes cover
    fractions, and this keeps every column summing to one exactly as the
    percentage path does.
    """
    codes = np.asarray(values)
    if codes.ndim != 2:  # pragma: no cover - defensive
        raise ValueError(f"dominant-class band must be 2-D, got {codes.shape}")
    codes = codes.astype(np.int32, copy=False)

    valid = (codes >= 1) & (codes <= N_AGGREGATE_CLASSES)
    unknown = np.unique(codes[~valid & (codes != DOMINANT_CLASS_NO_DATA)])
    if unknown.size:
        raise ValueError(
            f"{path.name} holds class code(s) {unknown.tolist()} outside "
            f"0-{N_AGGREGATE_CLASSES}; this is not an LCM aggregate-class raster"
        )

    fractions = np.zeros((N_AGGREGATE_CLASSES, *codes.shape), dtype=np.float32)
    rows, cols = np.nonzero(valid)
    fractions[codes[rows, cols] - 1, rows, cols] = 1.0
    return fractions


def _percentages_to_fractions(values: np.ndarray) -> np.ndarray:
    """Convert 0-100 percentage bands into per-parcel cover fractions.

    The published product rounds each band independently, so cover totals of up
    to 103% occur; those parcels are rescaled to sum to one.
    """
    fractions = np.clip(np.asarray(values, dtype=np.float32) / 100.0, 0.0, 1.0)
    totals = fractions.sum(axis=0, keepdims=True)
    scale = np.where(totals > 1.0, totals, 1.0)
    return (fractions / scale).astype(np.float32)


def bng_origin_to_index(
    raster: LCMRaster, easting_km: float, northing_km: float
) -> tuple[int, int]:
    """``(row, col)`` of the parcel whose *top-left* corner is at that point.

    The coordinates are the projected CRS's, in kilometres - British National
    Grid eastings and northings for the UKCEH products - and name the window's
    north-west corner, which is how a BNG extent is normally quoted ("E 495-545
    km, N 140-190 km" has its top-left at 495 / 190).
    """
    if raster.origin_xy is None:
        raise ValueError(
            f"{raster.path.name} is not georeferenced: a BNG origin needs a "
            "raster transform, so pass a (row, col) origin instead"
        )
    pixel_km = float(raster.pixel_size_m) / 1000.0
    origin_e_km = float(raster.origin_xy[0]) / 1000.0
    origin_n_km = float(raster.origin_xy[1]) / 1000.0

    col_float = (float(easting_km) - origin_e_km) / pixel_km
    row_float = (origin_n_km - float(northing_km)) / pixel_km
    row, col = round(row_float), round(col_float)
    off_grid = (
        abs(row_float - row) > _INDEX_TOLERANCE
        or abs(col_float - col) > _INDEX_TOLERANCE
    )
    if off_grid:
        raise ValueError(
            f"BNG origin ({easting_km}, {northing_km}) km does not fall on a "
            f"parcel boundary of a {pixel_km:g} km raster"
        )
    return int(row), int(col)


def _as_shape(shape: int | tuple[int, int]) -> tuple[int, int]:
    """``50`` or ``(rows, cols)`` -> ``(rows, cols)``."""
    if isinstance(shape, int):
        return int(shape), int(shape)
    rows, cols = shape
    return int(rows), int(cols)


def select_region(
    raster: LCMRaster,
    shape: int | tuple[int, int] = 50,
    origin: tuple[int, int] | None = None,
    bng_origin_km: tuple[float, float] | None = None,
) -> ParcelGrid:
    """Cut a ``rows x cols`` parcel window out of the raster.

    ``shape`` is either a single number (a square window, the thesis's 50 x 50)
    or an explicit ``(rows, columns)``, which is what the whole-region reading
    of the paper's Figure 1 needs.  The window may be pinned by raster
    ``origin = (row, col)`` or by ``bng_origin_km = (easting, northing)`` of its
    top-left corner; without either, the window with the most land cover is
    chosen, which keeps simulations away from the large sea / out-of-coverage
    areas of the source product.
    """
    rows, cols = _as_shape(shape)
    if rows < 1 or cols < 1:
        raise ValueError("a window must have positive extent")
    if rows > raster.height or cols > raster.width:
        raise ValueError(
            f"window {rows}x{cols} exceeds raster extent "
            f"{raster.height}x{raster.width}"
        )
    row, col = resolve_window_origin(raster, (rows, cols), origin, bng_origin_km)
    window = raster.fractions[:, row : row + rows, col : col + cols]
    return ParcelGrid.from_fractions(window)


def select_region_of_interest(
    raster: LCMRaster,
    grid_size: int = 50,
    origin: tuple[int, int] | None = None,
    bng_origin_km: tuple[float, float] | None = None,
) -> ParcelGrid:
    """Cut a ``grid_size x grid_size`` parcel window - :func:`select_region`'s
    square special case."""
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if grid_size > raster.height or grid_size > raster.width:
        raise ValueError(
            f"grid_size {grid_size} exceeds raster extent {raster.height}x{raster.width}"
        )
    return select_region(raster, grid_size, origin, bng_origin_km)


def resolve_window_origin(
    raster: LCMRaster,
    grid_size: int | tuple[int, int],
    origin: tuple[int, int] | None = None,
    bng_origin_km: tuple[float, float] | None = None,
) -> tuple[int, int]:
    """The ``(row, col)`` a window will actually be cut at, whichever form it came in.

    Split out of :func:`select_region` so a campaign run can *record* the window
    it used even when it was chosen automatically.  ``grid_size`` is a single
    number for a square window or ``(rows, cols)`` for a rectangular one.
    """
    rows, cols = _as_shape(grid_size)
    if origin is not None and bng_origin_km is not None:
        raise ValueError(
            "give a (row, col) origin or a BNG origin, not both: "
            f"{origin!r} and {bng_origin_km!r}"
        )
    if origin is None and bng_origin_km is None:
        return _densest_land_window(raster, (rows, cols))
    if bng_origin_km is not None:
        row, col = bng_origin_to_index(raster, *bng_origin_km)
        label = f"BNG origin {tuple(bng_origin_km)} km -> (row, col) ({row}, {col})"
    elif origin is not None:
        row, col = int(origin[0]), int(origin[1])
        label = f"origin {origin}"
    else:  # pragma: no cover - the branches above are exhaustive
        raise ValueError("no window origin given")
    if row < 0 or col < 0:
        raise ValueError(f"{label} must be non-negative")
    if row + rows > raster.height or col + cols > raster.width:
        raise ValueError(
            f"{label} with window {rows}x{cols} leaves the raster extent "
            f"{raster.height}x{raster.width}"
        )
    return row, col


def _densest_land_window(
    raster: LCMRaster, grid_size: int | tuple[int, int]
) -> tuple[int, int]:
    """Top-left corner of the window containing the most land cover."""
    rows, cols = _as_shape(grid_size)
    land = raster.fractions.sum(axis=0).astype(np.float64)
    integral = np.zeros((land.shape[0] + 1, land.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = land.cumsum(axis=0).cumsum(axis=1)
    totals = (
        integral[rows:, cols:]
        - integral[:-rows, cols:]
        - integral[rows:, :-cols]
        + integral[:-rows, :-cols]
    )
    flat_index = int(np.argmax(totals))
    return divmod(flat_index, totals.shape[1])
