"""Small shared helpers for neighbourhood operations on parcel grids."""

from __future__ import annotations

import math

import numpy as np

__all__ = ["kernel_weight", "neighbourhood_offsets", "shift"]


def neighbourhood_offsets(
    radius: int, include_centre: bool = True
) -> tuple[tuple[int, int], ...]:
    """Square-neighbourhood offsets in a fixed, deterministic order."""
    if radius < 1:
        raise ValueError("radius must be at least one parcel")
    return tuple(
        (row, col)
        for row in range(-radius, radius + 1)
        for col in range(-radius, radius + 1)
        if include_centre or (row, col) != (0, 0)
    )


def kernel_weight(row_offset: int, col_offset: int, decay_length: float) -> float:
    """Distance decay ``exp(-d / decay_length)`` for a neighbourhood offset."""
    if decay_length <= 0.0:
        raise ValueError("decay_length must be positive")
    return math.exp(-math.hypot(row_offset, col_offset) / decay_length)


def shift(array: np.ndarray, row_offset: int, col_offset: int) -> np.ndarray:
    """Zero-filled shift: ``result[i, j] = array[i + row_offset, j + col_offset]``."""
    if array.ndim != 2:
        raise ValueError("shift expects a 2-D array")
    shifted = np.zeros_like(array)
    height, width = array.shape
    src_row_start = max(row_offset, 0)
    src_row_end = min(height + row_offset, height)
    src_col_start = max(col_offset, 0)
    src_col_end = min(width + col_offset, width)
    if src_row_start >= src_row_end or src_col_start >= src_col_end:
        return shifted
    dst_row_start = src_row_start - row_offset
    dst_col_start = src_col_start - col_offset
    shifted[
        dst_row_start : dst_row_start + (src_row_end - src_row_start),
        dst_col_start : dst_col_start + (src_col_end - src_col_start),
    ] = array[src_row_start:src_row_end, src_col_start:src_col_end]
    return shifted
