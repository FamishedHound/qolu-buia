"""Per-parcel compliance with a top-down objective.

Section 3.5, "Reward Allocation and Iteration Completion":

    "Each QOLU agent then allocates rewards based on the degree of compliance
    with their individual objectives:  R_QOLU,i = f(parcel_data, objectives_i)"

``parcel_data`` is singular, so compliance is read here as a property of a
*single* candidate parcel under a *single* objective, on the same [0, 1] scale
the result tables use (1 = the parcel is exactly what that stakeholder wants
developed, 0 = the worst possible parcel for that stakeholder):

* ``LAND_USE_SEPARATION``   - normalised distance from the parcel to the
  nearest bog / mountain / heath parcel (further = better);
* ``WOODLAND_PRESERVATION`` - one minus the protected woodland + arable cover
  that converting this parcel would destroy;
* ``URBAN_PROXIMITY``       - one minus the normalised distance to the nearest
  parcel that is already urban (closer = better).

These are the same three definitions used by the iteration-level rewards in
:mod:`chapter4_qolu.env.rewards` and the run-level metrics in
:mod:`chapter4_qolu.env.evaluation`, evaluated for one parcel instead of a set,
so a stakeholder's per-parcel signal and its reported metric always agree.

Design note: compliance is measured against the landscape *before* the
synchronised update, i.e. the state the appraisal was made in.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from chapter4_qolu.config import RewardConfig
from chapter4_qolu.env.distances import max_distance, normalised_nearest_distance
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.land_cover import LandCoverGroup, group_indices
from chapter4_qolu.objectives import TopDownObjective

__all__ = ["compliance_scores", "parcel_compliance"]


def parcel_compliance(
    grid: ParcelGrid,
    parcel: ParcelIndex,
    objective: TopDownObjective,
    config: RewardConfig,
) -> float:
    """How well converting ``parcel`` complies with ``objective``, in [0, 1]."""
    if not grid.contains(parcel):
        raise IndexError(f"parcel {parcel} is outside the grid {grid.shape}")

    if objective is TopDownObjective.LAND_USE_SEPARATION:
        habitat = grid.group_mask(config.separation_group, config.presence_threshold)
        distance = normalised_nearest_distance(
            habitat, parcel.row, parcel.col, max_distance(grid, config)
        )
        # No habitat anywhere: nothing to keep away from.
        return 1.0 if distance is None else distance

    if objective is TopDownObjective.WOODLAND_PRESERVATION:
        indices: list[int] = []
        for group in config.protected_groups:
            indices.extend(group_indices(group))
        protected = float(
            grid.composition[sorted(set(indices)), parcel.row, parcel.col].sum()
        )
        return float(1.0 - min(max(protected, 0.0), 1.0))

    if objective is TopDownObjective.URBAN_PROXIMITY:
        urban = grid.group_mask(LandCoverGroup.URBAN, config.presence_threshold)
        distance = normalised_nearest_distance(
            urban, parcel.row, parcel.col, max_distance(grid, config)
        )
        # Nothing to connect to: no parcel can comply with this objective.
        return 0.0 if distance is None else float(1.0 - distance)

    raise ValueError(f"unknown objective: {objective!r}")  # pragma: no cover


def compliance_scores(
    grid: ParcelGrid,
    parcels: Sequence[ParcelIndex],
    objective: TopDownObjective,
    config: RewardConfig,
) -> np.ndarray:
    """:func:`parcel_compliance` for a whole decision buffer, in order."""
    return np.asarray(
        [parcel_compliance(grid, parcel, objective, config) for parcel in parcels],
        dtype=np.float64,
    ).reshape(-1)
