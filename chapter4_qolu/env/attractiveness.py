"""Attractiveness metric ``B_attractiveness = {A_1, ..., A_n}`` (Eq. 1).

The paper describes the metric qualitatively only; the concrete functional form
used here carries a design note on
:class:`chapter4_qolu.config.AttractivenessConfig`.

The field is deliberately expressed as a *local* neighbourhood average so that
"post-conversion, the attractiveness scores update to reflect changes in
neighbouring parcels" can be implemented as an exact incremental update rather
than a full recomputation: only parcels within ``radius`` of a converted parcel
can change.

Although attractiveness primarily drives the bottom-up investor agent (BUIA,
:mod:`chapter4_qolu.agent.buia_agent`), it is exposed as a standalone, reusable
object because the decision buffer sampling and the QOLU observations both
consume it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from chapter4_qolu.config import AttractivenessConfig
from chapter4_qolu.env.neighbourhood import kernel_weight, neighbourhood_offsets, shift
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex

__all__ = ["AttractivenessConfig", "AttractivenessField"]


@dataclass(frozen=True)
class AttractivenessField:
    """Per-parcel attractiveness scores in ``[0, 1]``."""

    scores: np.ndarray
    config: AttractivenessConfig

    def __post_init__(self) -> None:
        array = np.asarray(self.scores, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("attractiveness scores must be a 2-D array")
        array = array.copy()
        array.setflags(write=False)
        object.__setattr__(self, "scores", array)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_grid(
        cls, grid: ParcelGrid, config: AttractivenessConfig | None = None
    ) -> AttractivenessField:
        """Compute the attractiveness of every parcel from scratch."""
        config = config or AttractivenessConfig()
        features, weight_total = _feature_planes(grid, config)
        numerator = np.zeros(grid.shape, dtype=np.float64)
        denominator = np.zeros(grid.shape, dtype=np.float64)
        ones = np.ones(grid.shape, dtype=np.float64)
        for row_offset, col_offset in neighbourhood_offsets(config.radius):
            weight = kernel_weight(row_offset, col_offset, config.decay_length)
            numerator += weight * shift(features, row_offset, col_offset)
            denominator += weight * shift(ones, row_offset, col_offset)
        scores = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
        return cls(scores=scores / weight_total, config=config)

    # ------------------------------------------------------------------
    # incremental update
    # ------------------------------------------------------------------
    def updated(
        self, grid: ParcelGrid, changed: Iterable[ParcelIndex]
    ) -> AttractivenessField:
        """Return a new field after ``changed`` parcels were converted.

        Only parcels within ``config.radius`` of a changed parcel are
        recomputed; the result is identical to a full recomputation because the
        metric is a finite-support neighbourhood average.
        """
        changed = tuple(changed)
        if not changed:
            return self

        features, weight_total = _feature_planes(grid, self.config)
        scores = np.array(self.scores, dtype=np.float64, copy=True)
        radius = self.config.radius
        height, width = grid.shape

        affected: set[tuple[int, int]] = set()
        for parcel in changed:
            if not grid.contains(parcel):
                raise IndexError(f"parcel {parcel} is outside the grid {grid.shape}")
            for row in range(
                max(parcel.row - radius, 0), min(parcel.row + radius + 1, height)
            ):
                for col in range(
                    max(parcel.col - radius, 0), min(parcel.col + radius + 1, width)
                ):
                    affected.add((row, col))

        offsets = neighbourhood_offsets(radius)
        for row, col in sorted(affected):
            numerator = 0.0
            denominator = 0.0
            for row_offset, col_offset in offsets:
                neighbour_row = row + row_offset
                neighbour_col = col + col_offset
                if not (0 <= neighbour_row < height and 0 <= neighbour_col < width):
                    continue
                weight = kernel_weight(row_offset, col_offset, self.config.decay_length)
                numerator += weight * float(features[neighbour_row, neighbour_col])
                denominator += weight
            if denominator > 0.0:
                scores[row, col] = (numerator / denominator) / weight_total
            else:  # pragma: no cover - only reachable for an empty neighbourhood
                scores[row, col] = 0.0

        return AttractivenessField(scores=scores, config=self.config)

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def at(self, parcel: ParcelIndex) -> float:
        return float(self.scores[parcel.row, parcel.col])

    def patch(self, parcel: ParcelIndex, window: int) -> np.ndarray:
        """Zero-padded ``(window, window)`` patch centred on ``parcel``."""
        if window < 1 or window % 2 == 0:
            raise ValueError("window must be a positive odd number")
        radius = window // 2
        height, width = self.scores.shape
        patch = np.zeros((window, window), dtype=np.float64)
        row_start = max(parcel.row - radius, 0)
        row_end = min(parcel.row + radius + 1, height)
        col_start = max(parcel.col - radius, 0)
        col_end = min(parcel.col + radius + 1, width)
        patch[
            row_start - (parcel.row - radius) : row_end - (parcel.row - radius),
            col_start - (parcel.col - radius) : col_end - (parcel.col - radius),
        ] = self.scores[row_start:row_end, col_start:col_end]
        return patch


def _feature_planes(
    grid: ParcelGrid, config: AttractivenessConfig
) -> tuple[np.ndarray, float]:
    """Weighted sum of the desirable-feature cover planes and the weight total."""
    features = np.zeros(grid.shape, dtype=np.float64)
    total_weight = 0.0
    for group, weight in config.group_weights:
        features += weight * grid.group_fraction(group).astype(np.float64)
        total_weight += weight
    return features, total_weight
