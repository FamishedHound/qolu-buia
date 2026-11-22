"""Dominant-class maps of a parcel grid, and their compact JSON form.

A run record stores its initial and final landscapes as dominant-class maps
(:mod:`chapter4_qolu.run_record`).  The encoding lives here, beside the grid,
rather than in the record module, because the reporting layer reads those maps
back - to score a record on a metric it does not carry - and the record module
itself imports the reporting layer.  :mod:`chapter4_qolu.run_record` re-exports
every name.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from chapter4_qolu.env.parcel_grid import ParcelGrid

__all__ = ["GRID_NO_DATA", "class_grid", "decode_class_grid", "encode_class_grid"]

#: Value of a parcel that carries no land cover at all (sea / out of coverage).
#: The same sentinel :attr:`ParcelGrid.NO_DATA` uses.
GRID_NO_DATA: int = int(ParcelGrid.NO_DATA)


def class_grid(grid: ParcelGrid) -> np.ndarray:
    """Dominant-class map of a parcel grid, as 0-based ``land_cover`` indices.

    Design note (what a stored grid holds): the *dominant* aggregate class per
    parcel, indexed the way :mod:`chapter4_qolu.land_cover` indexes bands
    (0 = broadleaf woodland ... 9 = built-up), with :data:`GRID_NO_DATA` for a
    parcel with no cover at all.  The full ten-band composition would be forty
    times larger and is recoverable from the raster and the window anyway; the
    dominant class is what a land-cover map actually draws.
    """
    return np.asarray(grid.dominant_class(), dtype=np.int64)


def encode_class_grid(values: np.ndarray) -> dict[str, Any]:
    """Compact JSON form of a class grid: one comma-separated row per string.

    Design note (encoding): rows are stored as text rather than as nested lists
    of integers because every record is written with ``json.dumps(indent=2)``,
    which would put each of the 2,500 integers of a 50 x 50 grid on its own
    line - some 35 kB per grid.  One string per row keeps a 50 x 50 grid at
    about 5 kB and still round-trips exactly through
    :func:`decode_class_grid`.
    """
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError(f"a class grid must be 2-D, got shape {array.shape}")
    return {
        "shape": [int(array.shape[0]), int(array.shape[1])],
        "no_data": GRID_NO_DATA,
        "rows": [",".join(str(int(value)) for value in row) for row in array],
    }


def decode_class_grid(payload: Mapping[str, Any]) -> np.ndarray:
    """Inverse of :func:`encode_class_grid`."""
    rows = list(payload.get("rows") or [])
    shape = [int(value) for value in (payload.get("shape") or [len(rows), 0])]
    values = [[int(token) for token in row.split(",") if token != ""] for row in rows]
    array = np.array(values, dtype=np.int64) if values else np.zeros((0, 0), np.int64)
    if list(array.shape) != shape:
        raise ValueError(
            f"class grid shape {list(array.shape)} does not match the recorded "
            f"shape {shape}"
        )
    return array
