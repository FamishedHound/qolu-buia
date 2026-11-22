"""Distance helpers shared by the reward, compliance and evaluation modules.

All of the paper's proximity terms are "how far is this developed parcel from
the nearest parcel of type X", normalised so that the result is comparable
across grid sizes.  Design note: the normaliser is the grid diagonal unless
``RewardConfig.max_distance`` overrides it.
"""

from __future__ import annotations

import math

import numpy as np

from chapter4_qolu.config import RewardConfig
from chapter4_qolu.env.parcel_grid import ParcelGrid

__all__ = ["max_distance", "nearest_distance", "normalised_nearest_distance"]


def max_distance(grid: ParcelGrid, config: RewardConfig) -> float:
    """Distance used to normalise proximity terms into [0, 1]."""
    if config.max_distance is not None:
        return float(config.max_distance)
    return math.hypot(grid.height, grid.width)


def nearest_distance(mask: np.ndarray, row: int, col: int) -> float | None:
    """Euclidean distance from ``(row, col)`` to the nearest true cell."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return float(np.hypot(rows - row, cols - col).min())


def normalised_nearest_distance(
    mask: np.ndarray, row: int, col: int, limit: float
) -> float | None:
    """:func:`nearest_distance` divided by ``limit`` and clipped to [0, 1]."""
    distance = nearest_distance(mask, row, col)
    if distance is None:
        return None
    return float(min(distance / limit, 1.0))
