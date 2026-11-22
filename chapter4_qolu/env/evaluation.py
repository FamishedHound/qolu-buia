"""The paper's evaluation metrics (Section 4, "Overview" and Tables 4.4-4.7).

    (i)   Proximity to Desired Land Types - how close developed parcels are to
          bog, mountain and heath areas;
    (ii)  Preservation of Land Types - how much woodland and arable land
          survived;
    (iii) Urban Connectivity - how well developed parcels connect to existing
          urban areas.

Reported alongside the conversion-alignment metrics - the published
``conversion_alignment`` and the total-variation ``c_align`` (lower is better for
both; see :mod:`chapter4_qolu.env.metrics`) - and the well-being metric ``WB``.

Design note (orientation and normalisation): the result tables report every
score "normalized between 0 and 1" with the *best* method scoring highest, so
each metric here is built so that 1.0 is the outcome that stakeholder wants:

* ``land_use``   - mean distance of new development to the protected habitat,
  divided by the grid diagonal (further away = better, matching objective (i)
  which *maximises* that distance);
* ``woodland``   - share of the initial woodland + arable area still standing;
* ``urban``      - ``1 - distance / diagonal`` to the nearest pre-existing urban
  parcel (closer = better).

Design note: "developed parcels" are the parcels that crossed the built-up
presence threshold during the run, computed from the initial and final grids so
that the metrics can be applied to any run, including one that was not
instrumented.  A run with no new development returns 1.0 for the two distance
metrics (no badly-placed development to punish); this is stated explicitly
because it is a scoring convention, not a published rule.
"""

from __future__ import annotations

import numpy as np

from chapter4_qolu.config import RewardConfig, WellBeingConfig
from chapter4_qolu.env.attractiveness import AttractivenessField
from chapter4_qolu.env.distances import max_distance
from chapter4_qolu.env.metrics import c_align, conversion_alignment
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.env.well_being import well_being
from chapter4_qolu.land_cover import LandCoverGroup, group_indices

__all__ = [
    "distance_to_desired_land_types",
    "evaluate_landscape",
    "newly_developed_mask",
    "preservation_of_land_types",
    "urban_connectivity",
]


def newly_developed_mask(
    initial_grid: ParcelGrid, final_grid: ParcelGrid, threshold: float = 0.5
) -> np.ndarray:
    """Parcels that became urban between the two states."""
    if initial_grid.shape != final_grid.shape:
        raise ValueError(
            f"grid shapes differ: {initial_grid.shape} and {final_grid.shape}"
        )
    before = initial_grid.group_mask(LandCoverGroup.URBAN, threshold)
    after = final_grid.group_mask(LandCoverGroup.URBAN, threshold)
    return after & ~before


def _max_distance(grid: ParcelGrid, config: RewardConfig) -> float:
    return max_distance(grid, config)


def _mean_normalised_distance(
    developed: np.ndarray, targets: np.ndarray, limit: float
) -> float | None:
    """Mean distance from each developed parcel to the nearest target parcel."""
    target_rows, target_cols = np.nonzero(targets)
    if target_rows.size == 0:
        return None
    rows, cols = np.nonzero(developed)
    distances = [
        float(np.hypot(target_rows - row, target_cols - col).min())
        for row, col in zip(rows, cols)
    ]
    return float(np.mean([min(distance / limit, 1.0) for distance in distances]))


def distance_to_desired_land_types(
    initial_grid: ParcelGrid, final_grid: ParcelGrid, config: RewardConfig
) -> float:
    """Metric (i): distance of new development to bog / mountain / heath."""
    developed = newly_developed_mask(
        initial_grid, final_grid, config.presence_threshold
    )
    if not developed.any():
        return 1.0
    habitat = initial_grid.group_mask(
        config.separation_group, config.presence_threshold
    )
    score = _mean_normalised_distance(
        developed, habitat, _max_distance(initial_grid, config)
    )
    return 1.0 if score is None else score


def preservation_of_land_types(
    initial_grid: ParcelGrid, final_grid: ParcelGrid, config: RewardConfig
) -> float:
    """Metric (ii): share of the initial woodland + arable area still standing."""
    if initial_grid.shape != final_grid.shape:
        raise ValueError("grids must have the same shape")
    indices: list[int] = []
    for group in config.protected_groups:
        indices.extend(group_indices(group))
    indices = sorted(set(indices))
    initial_area = float(initial_grid.composition[indices].sum())
    if initial_area <= 0.0:
        return 1.0
    final_area = float(final_grid.composition[indices].sum())
    return float(min(final_area / initial_area, 1.0))


def urban_connectivity(
    initial_grid: ParcelGrid, final_grid: ParcelGrid, config: RewardConfig
) -> float:
    """Metric (iii): closeness of new development to pre-existing urban land."""
    developed = newly_developed_mask(
        initial_grid, final_grid, config.presence_threshold
    )
    if not developed.any():
        return 1.0
    urban = initial_grid.group_mask(LandCoverGroup.URBAN, config.presence_threshold)
    score = _mean_normalised_distance(
        developed, urban, _max_distance(initial_grid, config)
    )
    return 0.0 if score is None else float(1.0 - score)


def evaluate_landscape(
    initial_grid: ParcelGrid,
    final_grid: ParcelGrid,
    attractiveness: AttractivenessField,
    population: float,
    reward_config: RewardConfig | None = None,
    well_being_config: WellBeingConfig | None = None,
) -> dict[str, float]:
    """All five reported metrics for one run, keyed by result-table column."""
    reward_config = reward_config or RewardConfig()
    well_being_config = well_being_config or WellBeingConfig()
    return {
        "land_use": distance_to_desired_land_types(
            initial_grid, final_grid, reward_config
        ),
        "woodland": preservation_of_land_types(initial_grid, final_grid, reward_config),
        "urban": urban_connectivity(initial_grid, final_grid, reward_config),
        "conversion_alignment": conversion_alignment(initial_grid, final_grid),
        "c_align": c_align(initial_grid, final_grid),
        "well_being": well_being(
            final_grid, attractiveness, population, well_being_config
        ),
    }
