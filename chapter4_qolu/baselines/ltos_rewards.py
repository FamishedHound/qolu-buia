"""Reward decomposition and reward sharing for the LToS baseline.

LToS needs a *per-agent environment reward* ``r_i`` whose sum is the global
objective; the published Chapter 4 environment supplies a single shared team
reward.  The decomposition used here is the one the research note recommends:

    r_i = sum_obj w_obj * R_obj,i  +  R_society / K

where ``R_obj,i`` is the published objective term restricted to the conversions
that happened *inside* district ``i``, and ``K`` is the number of districts.

Two details make ``sum_i r_i`` exactly the published team reward:

* **Whole-map normalisers.**  Each published objective term is the *mean* over
  the conversions of a per-conversion score, so the per-district term divides by
  the **global** conversion count, not by the district's own.  Distances are
  likewise normalised by the whole-grid diagonal
  (:func:`chapter4_qolu.env.distances.max_distance`), never per district.  The
  note flags this explicitly: per-district normalisers would stop the sum
  matching the published objective.
* **The societal term.**  Design note: ``R_society`` is a whole-landscape
  quantity (the surviving share of forest and farmland) with no spatial
  attribution, so it is **split equally** across districts.  Adding it to every
  local reward - the other option the note lists - would multiply the global
  objective by ``K`` and destroy the unbiasedness that makes LToS a legitimate
  baseline against a team-reward method.  The same applies to the land
  preference term ``P_society`` of Eq. 3, which is not part of the environment's
  reward at all and is therefore simply not in the decomposition.

Reward *sharing* itself (Eq. 2 of Yi et al.) is
``r_i^w = sum_{j in N_i} w_ji r_j``; because every row of ``w`` sums to one over
``N_i`` and the lattice adjacency is symmetric, ``sum_i r_i^w = sum_i r_i``, so
sharing is conservative and the global objective stays unbiased.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from chapter4_qolu.baselines.districts import N_SLOTS, REVERSE_SLOT, DistrictLattice
from chapter4_qolu.config import RewardConfig, ScenarioWeights
from chapter4_qolu.env.parcel_grid import Conversion, ParcelGrid
from chapter4_qolu.env.rewards import (
    land_use_separation_reward,
    urban_proximity_reward,
    woodland_preservation_reward,
)
from chapter4_qolu.objectives import TopDownObjective

__all__ = ["district_rewards", "per_conversion_scores", "shaped_rewards"]

_SCORERS = {
    TopDownObjective.LAND_USE_SEPARATION: land_use_separation_reward,
    TopDownObjective.WOODLAND_PRESERVATION: woodland_preservation_reward,
    TopDownObjective.URBAN_PROXIMITY: urban_proximity_reward,
}


def per_conversion_scores(
    grid_before: ParcelGrid,
    conversions: Sequence[Conversion],
    config: RewardConfig,
) -> dict[TopDownObjective, np.ndarray]:
    """The three published objective terms, evaluated one conversion at a time.

    The functions in :mod:`chapter4_qolu.env.rewards` average over the
    conversions they are given, so calling each of them on a *single* conversion
    returns that conversion's own score.  Nothing is re-derived here, which is
    what keeps the decomposition provably consistent with the published reward.
    """
    conversions = tuple(conversions)
    return {
        objective: np.asarray(
            [scorer(grid_before, (conversion,), config) for conversion in conversions],
            dtype=np.float64,
        ).reshape(-1)
        for objective, scorer in _SCORERS.items()
    }


def district_rewards(
    lattice: DistrictLattice,
    grid_before: ParcelGrid,
    conversions: Sequence[Conversion],
    weights: ScenarioWeights,
    societal: float,
    config: RewardConfig,
) -> np.ndarray:
    """``r_i`` for every district; ``sum_i r_i`` is the published team reward."""
    conversions = tuple(conversions)
    rewards = np.full(
        lattice.n_districts, float(societal) / lattice.n_districts, dtype=np.float64
    )
    if not conversions:
        return rewards

    owners = np.asarray(
        [lattice.district_of(conversion.parcel) for conversion in conversions],
        dtype=np.int64,
    )
    scores = per_conversion_scores(grid_before, conversions, config)
    count = float(len(conversions))
    for objective, values in scores.items():
        weight = float(weights[objective])
        if weight == 0.0:
            continue
        np.add.at(rewards, owners, weight * values / count)
    return rewards


def shaped_rewards(
    local_rewards: np.ndarray, sharing_weights: np.ndarray, lattice: DistrictLattice
) -> np.ndarray:
    """``r_i^w = sum_{j in N_i} w_ji r_j`` (Eq. 2 of Yi et al., 2021).

    ``sharing_weights[j]`` is district ``j``'s *outgoing* vector ``w_j^out`` in
    the fixed slot layout, so the share ``j`` sends to ``i`` sits at the slot of
    ``j`` that points back at ``i``.
    """
    rewards = np.asarray(local_rewards, dtype=np.float64).reshape(-1)
    weights = np.asarray(sharing_weights, dtype=np.float64)
    if rewards.size != lattice.n_districts:
        raise ValueError(
            f"got {rewards.size} rewards for {lattice.n_districts} districts"
        )
    if weights.shape != (lattice.n_districts, N_SLOTS):
        raise ValueError(
            f"sharing weights must have shape "
            f"({lattice.n_districts}, {N_SLOTS}), got {weights.shape}"
        )

    shaped = np.zeros(lattice.n_districts, dtype=np.float64)
    for district in range(lattice.n_districts):
        total = 0.0
        for slot in range(N_SLOTS):
            if lattice.neighbour_mask[district, slot] == 0.0:
                continue
            neighbour = int(lattice.neighbour_index[district, slot])
            total += weights[neighbour, REVERSE_SLOT[slot]] * rewards[neighbour]
        shaped[district] = total
    return shaped
