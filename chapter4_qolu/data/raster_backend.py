"""Thin raster-reading abstraction.

``rasterio`` is preferred because it exposes CRS, affine transform and nodata
directly; ``tifffile`` (already a declared dependency of this repository) is
used as a fallback so the pipeline also runs in environments without GDAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

__all__ = ["RasterData", "Backend", "read_raster", "available_backends"]

Backend = Literal["auto", "rasterio", "tifffile"]

#: GeoTIFF tag holding (x, y, z) pixel sizes in CRS units.
_MODEL_PIXEL_SCALE_TAG = 33550
#: GeoTIFF tag holding raster <-> model tie points, six doubles per point:
#: ``(i, j, k, x, y, z)``.  The first point of an untransformed north-up image
#: is ``(0, 0, 0, easting, northing, 0)``, i.e. the top-left corner.
_MODEL_TIEPOINT_TAG = 33922


@dataclass(frozen=True)
class RasterData:
    """Raw band values plus the little geospatial metadata we need."""

    values: np.ndarray  # (bands, height, width)
    pixel_size_m: float | None
    crs: str | None
    backend: str
    #: Map coordinates ``(easting, northing)`` of the raster's top-left corner,
    #: in CRS units.  ``None`` when the file carries no transform.  Needed so a
    #: window can be pinned by British National Grid coordinates rather than by
    #: row and column.
    origin_xy: tuple[float, float] | None = None


def available_backends() -> tuple[str, ...]:
    """Raster backends importable in the current environment."""
    backends: list[str] = []
    for name in ("rasterio", "tifffile"):
        try:
            __import__(name)
        except ImportError:  # pragma: no cover - environment dependent
            continue
        backends.append(name)
    return tuple(backends)


def read_raster(path: Path, backend: Backend = "auto") -> RasterData:
    """Read a (multi-band) GeoTIFF into a ``(bands, height, width)`` array."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"raster not found: {path}")
    if backend not in ("auto", "rasterio", "tifffile"):
        raise ValueError(
            f"unknown raster backend {backend!r}; expected 'auto', 'rasterio' or 'tifffile'"
        )

    if backend in ("auto", "rasterio"):
        try:
            return _read_with_rasterio(path)
        except ImportError:
            if backend == "rasterio":
                raise ImportError(
                    "rasterio is not installed; install it or pass backend='tifffile'"
                ) from None

    try:
        return _read_with_tifffile(path)
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "no raster backend available: install rasterio (preferred) or tifffile"
        ) from error


def _read_with_rasterio(path: Path) -> RasterData:
    import rasterio  # imported lazily so tifffile-only environments still work

    with rasterio.open(path) as dataset:
        values = dataset.read()
        transform = dataset.transform
        pixel_size = abs(float(transform.a)) if transform else None
        origin = (float(transform.c), float(transform.f)) if transform else None
        crs = str(dataset.crs) if dataset.crs else None
    return RasterData(
        values=np.asarray(values),
        pixel_size_m=pixel_size,
        crs=crs,
        backend="rasterio",
        origin_xy=origin,
    )


def _read_with_tifffile(path: Path) -> RasterData:
    import tifffile

    with tifffile.TiffFile(path) as handle:
        page = handle.pages[0]
        values = page.asarray()
        pixel_size: float | None = None
        tag = page.tags.get(_MODEL_PIXEL_SCALE_TAG)
        if tag is not None and len(tag.value) >= 1:
            pixel_size = abs(float(tag.value[0]))
        origin = _tiepoint_origin(page.tags.get(_MODEL_TIEPOINT_TAG))

    array = np.asarray(values)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    elif array.ndim == 3:
        # tifffile returns (height, width, samples) for interleaved TIFFs.  The
        # copy back to C order matters: numpy's pairwise summation groups
        # differently on strided views, which would otherwise make band totals
        # differ from the rasterio path in the last float32 digit.
        array = np.ascontiguousarray(np.transpose(array, (2, 0, 1)))
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported raster shape {array.shape}")

    return RasterData(
        values=array,
        pixel_size_m=pixel_size,
        crs=None,
        backend="tifffile",
        origin_xy=origin,
    )


def _tiepoint_origin(tag: object) -> tuple[float, float] | None:
    """Top-left corner from a GeoTIFF ModelTiepoint tag, when it is usable.

    Design note: only the simple north-up case is read - one tie point that maps
    raster ``(0, 0)`` onto a map coordinate.  Anything else (rotated images,
    several tie points) is reported as "no origin" rather than guessed at; the
    rasterio backend, which is what the scan actually runs on, reads the full
    affine transform instead.
    """
    value = getattr(tag, "value", None)
    if value is None or len(value) < 6:
        return None
    raster_i, raster_j = float(value[0]), float(value[1])
    if raster_i != 0.0 or raster_j != 0.0:
        return None
    return float(value[3]), float(value[4])
