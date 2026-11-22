"""Optional polygon boundary for the whole-region reading of Figure 1.

The whole-region extent is a **bounding box** - BNG E 410-650 km, N 60-270 km -
not the administrative South East England region, whose boundary is a polygon.
The box therefore includes a strip of Hampshire, a corner of Wiltshire and some
sea; the sea is already excluded by the land mask, the neighbouring counties
are not.

When a boundary file *is* supplied (``--region-mask boundary.geojson`` or a
shapefile), this module rasterises it onto the window with
``rasterio.features.geometry_mask`` and the cells outside it are blanked - they
become all-zero columns, i.e. exactly what :meth:`ParcelGrid.land_mask` already
excludes, so no other code has to learn about regions at all.

Design note (no boundary bundled): none is downloaded and none is shipped.  A
run without ``--region-mask`` uses the bounding box, and the report says so in
as many words.  This keeps the package dependency-free and keeps the extent an
explicit, disclosed choice rather than a silent one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from chapter4_qolu.env.parcel_grid import ParcelGrid

__all__ = [
    "apply_region_mask",
    "load_region_mask",
    "region_mask_note",
]

logger = logging.getLogger(__name__)


def load_region_mask(
    boundary: Path | str | None,
    shape: tuple[int, int],
    transform: Any | None,
) -> np.ndarray | None:
    """Rasterise ``boundary`` onto a window, or ``None`` when there is none.

    Returns a boolean array of ``shape``, ``True`` inside the polygon.  Needs
    ``rasterio`` (both to read the vector file and to rasterise it); without it
    the call raises rather than silently returning the whole box, because a run
    that asked for a boundary and did not get one is not the run that was
    asked for.
    """
    if boundary is None:
        return None
    path = Path(boundary)
    if not path.is_file():
        raise FileNotFoundError(f"region boundary not found: {path}")
    if transform is None:
        raise ValueError(
            f"{path.name} cannot be rasterised: the raster window has no "
            "affine transform, so there is nothing to project the polygon onto"
        )
    try:
        import fiona  # type: ignore[import-not-found]
        import rasterio.features  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - depends on the interpreter
        raise ImportError(
            "--region-mask needs rasterio (and fiona) to read and rasterise a "
            f"boundary file; {error}"
        ) from error

    with fiona.open(path) as source:
        geometries = [feature["geometry"] for feature in source]
    if not geometries:
        raise ValueError(f"{path.name} holds no geometry")
    inside = rasterio.features.geometry_mask(
        geometries, out_shape=shape, transform=transform, invert=True
    )
    logger.info(
        "region mask %s: %d of %d cells inside the boundary",
        path.name,
        int(inside.sum()),
        inside.size,
    )
    return np.asarray(inside, dtype=bool)


def apply_region_mask(grid: ParcelGrid, mask: np.ndarray | None) -> ParcelGrid:
    """Blank every parcel outside ``mask``, leaving the rest untouched.

    A blanked parcel has zero cover in every class, which is how the source
    product already spells "sea / outside the mapped area", so it is excluded
    by :meth:`ParcelGrid.land_mask` and is neither convertible nor developable.
    """
    if mask is None:
        return grid
    inside = np.asarray(mask, dtype=bool)
    if inside.shape != grid.shape:
        raise ValueError(
            f"region mask shape {inside.shape} does not match the grid " f"{grid.shape}"
        )
    composition = np.array(grid.composition, dtype=np.float32, copy=True)
    composition[:, ~inside] = 0.0
    return ParcelGrid.from_fractions(composition)


def region_mask_note(boundary: Path | str | None) -> str:
    """One sentence for a report: was a boundary used, or just the box?"""
    if boundary is None:
        return (
            "No region boundary was supplied, so the extent is the **bounding "
            "box** BNG E 410-650 km, N 60-270 km and not the administrative "
            "South East England region: the box also contains parts of "
            "Hampshire, Wiltshire, Oxfordshire and Buckinghamshire, plus sea "
            "cells that the land mask already excludes."
        )
    return (
        f"The extent was clipped to the polygon boundary `{Path(boundary).name}`; "
        "cells outside it were blanked and are therefore neither convertible "
        "nor developable, exactly as sea cells are."
    )
