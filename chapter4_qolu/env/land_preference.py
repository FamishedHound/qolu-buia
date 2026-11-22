"""Land preference metric (Eq. 3 of the paper).

    LP = sum_i w_i * P_phi_TD_i(L) + P_society(L)

``P_phi_TD_i`` is the land preference distribution of the i-th top-down agent
and ``P_society`` is "the societal land preference function taken from a
dataset", which the paper visualises as per-land-type pie charts of the
"natural co-occurrence of land types in the real world".

Design note: ``P_society`` is therefore derived from the neighbourhood
co-occurrence of land types in the source raster - for every parcel of land
type ``a`` we accumulate the cover composition of its neighbours, and row
normalise.  That reproduces the published pie charts ("what tends to lie next
to this land type") directly from the UKCEH data.

Design note: the per-stakeholder distributions ``P_phi_TD_i`` are not given
numerically in the paper, so each objective is mapped onto a uniform
distribution over the land types that objective protects (see
:func:`objective_preference`).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from chapter4_qolu.config import ScenarioWeights
from chapter4_qolu.env.neighbourhood import neighbourhood_offsets, shift
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.land_cover import (
    N_AGGREGATE_CLASSES,
    AggregateClass,
    LandCoverGroup,
    band_index,
    group_indices,
)
from chapter4_qolu.objectives import TopDownObjective

__all__ = [
    "cooccurrence_matrix",
    "land_preference",
    "objective_preference",
    "societal_preference",
]

#: Land types each top-down objective wants to see preserved on the map.
_OBJECTIVE_GROUPS: Mapping[TopDownObjective, tuple[LandCoverGroup, ...]] = {
    TopDownObjective.LAND_USE_SEPARATION: (LandCoverGroup.MOUNTAIN_HEATH_BOG,),
    TopDownObjective.WOODLAND_PRESERVATION: (
        LandCoverGroup.WOODLAND,
        LandCoverGroup.ARABLE,
    ),
    TopDownObjective.URBAN_PROXIMITY: (LandCoverGroup.URBAN,),
}


def cooccurrence_matrix(grid: ParcelGrid, radius: int = 1) -> np.ndarray:
    """Row-stochastic land type co-occurrence matrix of a landscape.

    ``matrix[a, b]`` is the share of neighbouring cover of type ``b`` around
    parcels whose dominant land type is ``a``.  Rows for absent land types stay
    all zero.
    """
    if radius < 1:
        raise ValueError("radius must be at least one parcel")

    dominant = grid.dominant_class()
    neighbour_cover = np.zeros(grid.composition.shape, dtype=np.float64)
    for row_offset, col_offset in neighbourhood_offsets(radius, include_centre=False):
        for band in range(grid.n_classes):
            neighbour_cover[band] += shift(
                grid.composition[band].astype(np.float64), row_offset, col_offset
            )

    matrix = np.zeros((N_AGGREGATE_CLASSES, N_AGGREGATE_CLASSES), dtype=np.float64)
    for land_class in range(N_AGGREGATE_CLASSES):
        selector = dominant == land_class
        if not selector.any():
            continue
        matrix[land_class] = neighbour_cover[:, selector].sum(axis=1)

    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0.0)


def societal_preference(
    matrix: np.ndarray, focal_class: AggregateClass | None = None
) -> np.ndarray:
    """``P_society(L)``: the co-occurrence row of ``focal_class``.

    With ``focal_class=None`` the marginal (landscape-wide) co-occurrence
    distribution is returned instead.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (N_AGGREGATE_CLASSES, N_AGGREGATE_CLASSES):
        raise ValueError(
            f"co-occurrence matrix must be {N_AGGREGATE_CLASSES}x{N_AGGREGATE_CLASSES}"
        )
    if focal_class is not None:
        return np.array(matrix[band_index(focal_class)], dtype=np.float64)
    marginal = matrix.sum(axis=0)
    total = float(marginal.sum())
    if total <= 0.0:
        return np.zeros(N_AGGREGATE_CLASSES, dtype=np.float64)
    return marginal / total


def objective_preference(objective: TopDownObjective) -> np.ndarray:
    """``P_phi_TD_i(L)`` for one top-down objective (uniform over its land types)."""
    indices: list[int] = []
    for group in _OBJECTIVE_GROUPS[objective]:
        indices.extend(group_indices(group))
    preference = np.zeros(N_AGGREGATE_CLASSES, dtype=np.float64)
    preference[indices] = 1.0 / len(indices)
    return preference


def land_preference(
    preferences: Mapping[TopDownObjective, np.ndarray],
    weights: ScenarioWeights,
    societal: np.ndarray,
    normalise: bool = False,
) -> np.ndarray:
    """Combine stakeholder and societal land preferences.

    Follows Eq. 3 literally, i.e. the societal term is *added* with unit weight
    and the result is not renormalised unless ``normalise`` is requested.
    """
    societal_array = np.asarray(societal, dtype=np.float64)
    if societal_array.ndim != 1:
        raise ValueError("societal preference must be a 1-D distribution")

    total = np.array(societal_array, dtype=np.float64, copy=True)
    for objective in TopDownObjective:
        preference = np.asarray(preferences[objective], dtype=np.float64)
        if preference.shape != societal_array.shape:
            raise ValueError(
                f"preference for {objective} has shape {preference.shape}, "
                f"expected {societal_array.shape}"
            )
        total += weights[objective] * preference

    if normalise:
        magnitude = float(total.sum())
        if magnitude <= 0.0:
            raise ValueError("cannot normalise an all-zero land preference")
        total = total / magnitude
    return total
