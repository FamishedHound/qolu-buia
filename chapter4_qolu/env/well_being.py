"""Well-being metric (Section 3.6).

    WB = H_occupied / H_total

"The well-being metric considers the housing capacity and unoccupied houses
[...] assessing how effectively the housing capacity is utilized."

The paper gives the ratio but no housing model, so the minimal dynamics used
here carry design notes on :class:`chapter4_qolu.config.WellBeingConfig`:
capacity is proportional to built-up cover, and households choose parcels in
proportion to attractiveness in a single allocation pass.
"""

from __future__ import annotations

import numpy as np

from chapter4_qolu.config import WellBeingConfig
from chapter4_qolu.env.attractiveness import AttractivenessField
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.land_cover import LandCoverGroup

__all__ = [
    "housing_capacity",
    "occupied_housing",
    "population_at",
    "total_capacity",
    "well_being",
]


def housing_capacity(grid: ParcelGrid, config: WellBeingConfig) -> np.ndarray:
    """Homes available in each parcel, proportional to its built-up cover."""
    urban = grid.group_fraction(LandCoverGroup.URBAN).astype(np.float64)
    return urban * config.capacity_per_parcel


def total_capacity(grid: ParcelGrid, config: WellBeingConfig) -> float:
    """``H_total`` for the whole landscape."""
    return float(housing_capacity(grid, config).sum())


def population_at(
    iteration: int, initial_capacity: float, config: WellBeingConfig
) -> float:
    """Households seeking housing at ``iteration`` of a run."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    base = config.initial_occupancy * float(initial_capacity)
    return base * (1.0 + config.population_growth_per_iteration * iteration)


def occupied_housing(
    grid: ParcelGrid,
    attractiveness: AttractivenessField,
    population: float,
    config: WellBeingConfig,
) -> np.ndarray:
    """``H_occupied`` per parcel after one attractiveness-driven allocation pass."""
    if population < 0.0:
        raise ValueError("population must be non-negative")
    capacity = housing_capacity(grid, config)
    scores = np.asarray(attractiveness.scores, dtype=np.float64)
    if scores.shape != capacity.shape:
        raise ValueError(
            f"attractiveness {scores.shape} does not match the grid {capacity.shape}"
        )

    has_homes = capacity > 0.0
    if not has_homes.any() or population <= 0.0:
        return np.zeros_like(capacity)

    weights = np.where(has_homes, scores, 0.0)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        # No parcel is more attractive than any other: spread demand evenly.
        weights = has_homes.astype(np.float64)
        total_weight = float(weights.sum())

    demand = population * weights / total_weight
    return np.minimum(demand, capacity)


def well_being(
    grid: ParcelGrid,
    attractiveness: AttractivenessField,
    population: float,
    config: WellBeingConfig,
) -> float:
    """``WB = H_occupied / H_total`` in [0, 1] (0 when there is no housing)."""
    capacity_total = total_capacity(grid, config)
    if capacity_total <= 0.0:
        return 0.0
    occupied = occupied_housing(grid, attractiveness, population, config)
    return float(min(occupied.sum() / capacity_total, 1.0))
