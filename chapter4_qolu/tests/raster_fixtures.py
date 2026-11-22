"""Synthetic LCM rasters for the loader and window-scan tests.

Nothing here needs the real UKCEH products: a handful of class codes written to
a British-National-Grid GeoTIFF exercises the dominant-class path, the BNG
window selection and the window grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

__all__ = [
    "SYNTHETIC_ORIGIN_E_M",
    "SYNTHETIC_ORIGIN_N_M",
    "SYNTHETIC_PIXEL_M",
    "write_geotiff",
]

#: Top-left corner of the synthetic rasters, in British National Grid metres.
SYNTHETIC_ORIGIN_E_M = 400_000.0
SYNTHETIC_ORIGIN_N_M = 200_000.0
SYNTHETIC_PIXEL_M = 1000.0


def write_geotiff(
    path: Path,
    values: np.ndarray,
    origin_e_m: float = SYNTHETIC_ORIGIN_E_M,
    origin_n_m: float = SYNTHETIC_ORIGIN_N_M,
    pixel_m: float = SYNTHETIC_PIXEL_M,
) -> Path:
    """Write 2-D or ``(bands, height, width)`` values as a BNG GeoTIFF."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    array = np.asarray(values)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    transform = from_origin(origin_e_m, origin_n_m, pixel_m, pixel_m)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=array.dtype,
        crs="EPSG:27700",
        transform=transform,
    ) as dataset:
        dataset.write(array)
    return path
