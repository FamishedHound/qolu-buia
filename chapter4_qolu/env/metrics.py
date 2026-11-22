"""Conversion-alignment metrics (Section 4, "Conversion Metric").

Two definitions live here, on one neighbourhood.

The **published** one, :func:`conversion_alignment`, which every run record and
the per-iteration series carry:

    C_align(s) = sum_i ( commonality(s_i) - commonality(s_i_hat) )^2 / N

where ``s_i`` is the set of land types neighbouring parcel ``i`` before the
agents act, ``s_i_hat`` the set afterwards, and ``commonality`` "measures the
frequency of the most common neighbouring land type to parcel i".

The **total-variation** one, :func:`c_align`, reported under the
record key ``c_align``: the mean total-variation distance between each
parcel's neighbourhood land-type distribution before and after,

    C_align(s, s_hat) = 1/|P| sum_j 1/2 sum_c | p_jc - p_hat_jc |,

with ``p_jc`` the share of the parcels adjacent to ``j`` whose land type is
``c``.  It lies in [0, 1], lower is better, is 0 only when every neighbourhood
keeps its distribution, and a parcel's term is 1 when its neighbourhood shares
no land type with its previous one.  The published form compares only the
frequency of the modal type, so a neighbourhood going from six of eight
woodland to six of eight urban scores 0 there and 0.75 here.

Design note (the neighbourhood, shared by both): the eight adjacent parcels,
the focal parcel excluded, fewer at the border; a parcel's land type is its
dominant class.  Shares are relative to the number of valid neighbours, so the
metrics are invariant to how many neighbours a parcel happens to have at the
border.  Parcels without land cover ("no data", e.g. sea outside the LCM
coverage) are excluded from the neighbour counts instead of being treated as a
land type of their own, and the mean runs over every parcel of the grid.
"""

from __future__ import annotations

import numpy as np

from chapter4_qolu.env.neighbourhood import neighbourhood_offsets, shift
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex

__all__ = [
    "c_align",
    "c_align_from_classes",
    "c_align_map",
    "commonality",
    "commonality_map",
    "conversion_alignment",
]


def commonality_map(dominant_classes: np.ndarray, radius: int = 1) -> np.ndarray:
    """Relative frequency of the modal neighbouring land type, per parcel."""
    if radius < 1:
        raise ValueError("radius must be at least one parcel")
    classes = np.asarray(dominant_classes)
    if classes.ndim != 2:
        raise ValueError("dominant_classes must be a 2-D array")

    valid = classes >= 0
    present = (
        np.unique(classes[valid]) if valid.any() else np.empty(0, dtype=classes.dtype)
    )

    if present.size == 0:
        return np.zeros(classes.shape, dtype=np.float64)

    valid_plane = valid.astype(np.float64)
    class_planes = [
        ((classes == land_class) & valid).astype(np.float64) for land_class in present
    ]

    counts = np.zeros((present.size,) + classes.shape, dtype=np.float64)
    total = np.zeros(classes.shape, dtype=np.float64)
    # The focal parcel is not its own neighbour, hence include_centre=False.
    for row_offset, col_offset in neighbourhood_offsets(radius, include_centre=False):
        total += shift(valid_plane, row_offset, col_offset)
        for index, plane in enumerate(class_planes):
            counts[index] += shift(plane, row_offset, col_offset)

    modal = counts.max(axis=0)
    return np.divide(modal, total, out=np.zeros_like(modal), where=total > 0.0)


def commonality(
    dominant_classes: np.ndarray, parcel: ParcelIndex, radius: int = 1
) -> float:
    """Relative frequency of the most common land type neighbouring ``parcel``."""
    scores = commonality_map(dominant_classes, radius=radius)
    if not (0 <= parcel.row < scores.shape[0] and 0 <= parcel.col < scores.shape[1]):
        raise IndexError(f"parcel {parcel} is outside the grid {scores.shape}")
    return float(scores[parcel.row, parcel.col])


def conversion_alignment(
    before: ParcelGrid, after: ParcelGrid, radius: int = 1
) -> float:
    """Mean squared change in neighbourhood commonality (lower is better)."""
    if before.shape != after.shape:
        raise ValueError(
            f"grid shapes differ: {before.shape} before, {after.shape} after"
        )
    before_scores = commonality_map(before.dominant_class(), radius=radius)
    after_scores = commonality_map(after.dominant_class(), radius=radius)
    return float(np.mean((before_scores - after_scores) ** 2))


def _neighbour_counts(
    classes: np.ndarray, land_classes: np.ndarray, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class neighbour counts ``(len(land_classes), H, W)`` and their total."""
    valid = classes >= 0
    counts = np.zeros((land_classes.size,) + classes.shape, dtype=np.float64)
    total = np.zeros(classes.shape, dtype=np.float64)
    planes = [
        ((classes == land_class) & valid).astype(np.float64)
        for land_class in land_classes
    ]
    valid_plane = valid.astype(np.float64)
    # The focal parcel is not its own neighbour, hence include_centre=False.
    for row_offset, col_offset in neighbourhood_offsets(radius, include_centre=False):
        total += shift(valid_plane, row_offset, col_offset)
        for index, plane in enumerate(planes):
            counts[index] += shift(plane, row_offset, col_offset)
    return counts, total


def c_align_map(
    before_classes: np.ndarray, after_classes: np.ndarray, radius: int = 1
) -> np.ndarray:
    """Per-parcel total-variation distance of the neighbourhood distributions.

    Both arguments are dominant-class maps (``ParcelGrid.dominant_class()``, or
    the grids a run record stores), negative where a parcel has no land cover.
    A parcel with no land-covered neighbour has no distribution and scores 0.
    """
    if radius < 1:
        raise ValueError("radius must be at least one parcel")
    before = np.asarray(before_classes)
    after = np.asarray(after_classes)
    if before.ndim != 2 or before.shape != after.shape:
        raise ValueError(
            f"class maps must be 2-D and the same shape, got {before.shape} "
            f"and {after.shape}"
        )
    if not np.array_equal(before >= 0, after >= 0):
        raise ValueError(
            "the two landscapes must have the same land-covered parcels; "
            "conversion changes a parcel's type, never whether it has one"
        )
    present = np.union1d(before[before >= 0], after[after >= 0])
    if present.size == 0:
        return np.zeros(before.shape, dtype=np.float64)
    counts_before, total = _neighbour_counts(before, present, radius)
    counts_after, _ = _neighbour_counts(after, present, radius)
    distance = 0.5 * np.abs(counts_before - counts_after).sum(axis=0)
    return np.divide(distance, total, out=np.zeros_like(distance), where=total > 0.0)


def c_align_from_classes(
    before_classes: np.ndarray, after_classes: np.ndarray, radius: int = 1
) -> float:
    """``C_align`` of two dominant-class maps: the mean of :func:`c_align_map`."""
    return float(np.mean(c_align_map(before_classes, after_classes, radius=radius)))


def c_align(before: ParcelGrid, after: ParcelGrid, radius: int = 1) -> float:
    """The total-variation conversion metric of two landscapes (lower is better)."""
    if before.shape != after.shape:
        raise ValueError(
            f"grid shapes differ: {before.shape} before, {after.shape} after"
        )
    return c_align_from_classes(
        before.dominant_class(), after.dominant_class(), radius=radius
    )
