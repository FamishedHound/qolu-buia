"""Reward terms of the POSG (Section 3.1 and the experimental settings).

    R(s, a) = sum_i w_i R_phi_i(s, a) + R_society(s, a)

The environment supplies **one shared team reward** even though each top-down
agent computes its own objective, which is what makes the setting fully
cooperative in the paper.

The three stakeholder objectives are the ones the paper evaluates:

    (i)   distance of new development to bog / mountain / heath   (Land Use)
    (ii)  loss of woodland and arable land                        (Woodland)
    (iii) distance of new development to existing urban land      (Urban)

Design note: the paper reports all three metrics "normalized between 0 and 1",
so each term here is built as a score in [0, 1] where 1 is the best outcome for
that stakeholder, and distances are normalised by the grid diagonal.

Design note: an iteration in which *no* conversion is applied has no development
outcome to score, so the three per-conversion terms are 0.0 while the societal
term still rewards the preserved landscape.  Note that without the bottom-up
investor there is no housing-demand pressure, so a purely rejecting policy is
not penalised; supplying that pressure is exactly the role of the bottom-up
agent (BUIA) in the full system.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from chapter4_qolu.config import RewardConfig, ScenarioWeights
from chapter4_qolu.env.distances import max_distance, nearest_distance
from chapter4_qolu.env.parcel_grid import Conversion, ParcelGrid
from chapter4_qolu.land_cover import LandCoverGroup, group_indices
from chapter4_qolu.objectives import TopDownObjective

__all__ = [
    "RewardConfig",
    "TopDownObjective",
    "compose_team_reward",
    "land_use_separation_reward",
    "objective_rewards",
    "societal_reward",
    "urban_proximity_reward",
    "woodland_preservation_reward",
]


def _max_distance(grid: ParcelGrid, config: RewardConfig) -> float:
    return max_distance(grid, config)


def _nearest_distance(mask: np.ndarray, row: int, col: int) -> float | None:
    """Euclidean distance from ``(row, col)`` to the nearest true cell."""
    return nearest_distance(mask, row, col)


def land_use_separation_reward(
    grid_before: ParcelGrid,
    conversions: Sequence[Conversion],
    config: RewardConfig,
) -> float:
    """Objective (i): keep new development away from fragile habitat.

    Returns the mean normalised distance from each converted parcel to the
    nearest parcel of the protected habitat group; 1.0 when the habitat is
    absent from the landscape entirely.
    """
    if not conversions:
        return 0.0
    mask = grid_before.group_mask(config.separation_group, config.presence_threshold)
    limit = _max_distance(grid_before, config)
    scores = []
    for conversion in conversions:
        distance = _nearest_distance(mask, conversion.parcel.row, conversion.parcel.col)
        if distance is None:
            scores.append(1.0)
        else:
            scores.append(min(distance / limit, 1.0))
    return float(np.mean(scores))


def woodland_preservation_reward(
    grid_before: ParcelGrid,
    conversions: Sequence[Conversion],
    config: RewardConfig,
) -> float:
    """Objective (ii): minimise the woodland / arable cover lost to development."""
    if not conversions:
        return 0.0
    protected: list[int] = []
    for group in config.protected_groups:
        protected.extend(group_indices(group))
    protected = sorted(set(protected))
    scores = []
    for conversion in conversions:
        lost = float(
            np.asarray(conversion.lost_cover, dtype=np.float64)[protected].sum()
        )
        scores.append(1.0 - min(max(lost, 0.0), 1.0))
    return float(np.mean(scores))


def urban_proximity_reward(
    grid_before: ParcelGrid,
    conversions: Sequence[Conversion],
    config: RewardConfig,
) -> float:
    """Objective (iii): keep new development close to existing urban land.

    The urban mask is taken from the state *before* the synchronised update, so
    a conversion cannot reward itself.
    """
    if not conversions:
        return 0.0
    mask = grid_before.group_mask(LandCoverGroup.URBAN, config.presence_threshold)
    limit = _max_distance(grid_before, config)
    scores = []
    for conversion in conversions:
        distance = _nearest_distance(mask, conversion.parcel.row, conversion.parcel.col)
        if distance is None:
            scores.append(0.0)
        else:
            scores.append(1.0 - min(distance / limit, 1.0))
    return float(np.mean(scores))


def societal_reward(
    initial_grid: ParcelGrid, current_grid: ParcelGrid, config: RewardConfig
) -> float:
    """``R_society``: net public benefit of the remaining forest / farmland.

    ``alpha * remaining_forest_share + beta * remaining_agricultural_share``,
    each share measured against the landscape the run started from.
    """
    if initial_grid.shape != current_grid.shape:
        raise ValueError("initial and current grids must have the same shape")
    forest = _remaining_share(initial_grid, current_grid, config.societal_forest_group)
    agricultural = _remaining_share(
        initial_grid, current_grid, config.societal_agricultural_group
    )
    return float(config.alpha * forest + config.beta * agricultural)


def _remaining_share(
    initial_grid: ParcelGrid, current_grid: ParcelGrid, group: LandCoverGroup
) -> float:
    initial_area = float(initial_grid.group_fraction(group).sum())
    if initial_area <= 0.0:
        # A land type that never existed cannot be lost; treat as fully preserved.
        return 1.0
    current_area = float(current_grid.group_fraction(group).sum())
    return min(current_area / initial_area, 1.0)


def objective_rewards(
    grid_before: ParcelGrid,
    grid_after: ParcelGrid,
    conversions: Sequence[Conversion],
    config: RewardConfig,
) -> dict[TopDownObjective, float]:
    """All three stakeholder rewards ``R_phi_i`` for one synchronised update."""
    if grid_before.shape != grid_after.shape:
        raise ValueError("grids must have the same shape")
    return {
        TopDownObjective.LAND_USE_SEPARATION: land_use_separation_reward(
            grid_before, conversions, config
        ),
        TopDownObjective.WOODLAND_PRESERVATION: woodland_preservation_reward(
            grid_before, conversions, config
        ),
        TopDownObjective.URBAN_PROXIMITY: urban_proximity_reward(
            grid_before, conversions, config
        ),
    }


def compose_team_reward(
    rewards: Mapping[TopDownObjective, float],
    weights: ScenarioWeights,
    societal: float,
) -> float:
    """``sum_i w_i R_phi_i + R_society`` - the single shared team reward."""
    total = float(societal)
    for objective in TopDownObjective:
        total += float(weights[objective]) * float(rewards[objective])
    return total
