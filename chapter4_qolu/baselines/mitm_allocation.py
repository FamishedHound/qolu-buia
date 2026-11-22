"""MitM's attractivity surface, bottom-up chooser and MODM allocation.

Everything in this module is **INFERRED**: Bone, Dragicevic & White (2011) is
paywalled, and the implementation uses none of its figures, equations or
hyper-parameters.  It follows the research note, which pins the method down
from the abstract plus the paper's 45-item reference list:

* **White & Engelen (1997)** is cited, so the attractivity surface takes their
  constrained-CA form
  ``A_k(c) = s_k(c) * (1 + sum_d sum_k' w_{k,k',d} n_{k',d}(c)) * (1 + eps)``.
  Here the whole neighbourhood double sum is supplied by the published Eq. 1
  attractiveness field (:mod:`chapter4_qolu.env.attractiveness`), which the note
  identifies as the environment's own analogue of MitM's attractivity, rather
  than being re-derived per distance ring.  ``s_k`` is the multi-criteria
  suitability built from the stakeholders' criterion weights ``theta``.
* **Benenson (1998)** is cited for multi-agent residential dynamics, so the
  bottom-up layer is a *myopic* utility maximiser over ``A(c) * f(density)``
  that never learns.  Using the neural BUIA instead would turn MitM into
  QOLU + BUIA and destroy the comparison.
* **Cohon (1978)** and **Eastman & Jiang (1998)** are cited, so the conflict
  between stakeholders is resolved by compromise programming / MOLA: rank the
  alternatives by their weighted ``L_p`` distance to the ideal point and
  allocate the best ``B``.

Design note: ``p = 2`` (the Euclidean compromise) and the ideal / anti-ideal
points are taken over the *current* candidate set, which is what MOLA does when
the alternative set changes each round.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from chapter4_qolu.config import RewardConfig
from chapter4_qolu.env.compliance import parcel_compliance
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.land_cover import LandCoverGroup
from chapter4_qolu.objectives import TopDownObjective

__all__ = [
    "compromise_allocation",
    "compromise_scores",
    "criterion_matrix",
    "density_discount",
    "greedy_choice",
    "white_engelen_attractivity",
]

#: Window used for the local population-density term.  Design note: 3 x 3, the
#: immediate neighbourhood, matching the scale at which White & Engelen's
#: nearest distance ring operates.
_DENSITY_WINDOW: int = 3


def white_engelen_attractivity(
    suitability: np.ndarray,
    neighbourhood: np.ndarray,
    neighbourhood_weight: float,
    perturbation: np.ndarray,
) -> np.ndarray:
    """``A(c) = s(c) * (1 + lambda * N(c)) * (1 + eps)``."""
    return (
        np.asarray(suitability, dtype=np.float64)
        * (1.0 + float(neighbourhood_weight) * np.asarray(neighbourhood, np.float64))
        * (1.0 + np.asarray(perturbation, dtype=np.float64))
    )


def criterion_matrix(
    grid: ParcelGrid, parcels: Sequence[ParcelIndex], config: RewardConfig
) -> np.ndarray:
    """The MCE layers: one column per stakeholder criterion, in [0, 1].

    The columns are the published per-parcel compliance scores, in
    :class:`~chapter4_qolu.objectives.TopDownObjective` order, so MitM's
    criteria and the reported metrics are the same quantities.
    """
    return np.asarray(
        [
            [
                parcel_compliance(grid, parcel, objective, config)
                for objective in TopDownObjective
            ]
            for parcel in parcels
        ],
        dtype=np.float64,
    ).reshape(len(parcels), len(TopDownObjective))


def density_discount(
    grid: ParcelGrid, parcels: Sequence[ParcelIndex], pressure: float
) -> np.ndarray:
    """``f(population density)``: households avoid already-crowded cells.

    Design note: the abstract says agents "select locations to inhabit based on
    population densities and attractivity preferences" but gives no functional
    form, so the simplest monotone discount is used,
    ``f = 1 / (1 + pressure * built_up_share)``, with the built-up share of the
    3 x 3 neighbourhood standing in for population density (the environment has
    no population raster, only housing capacity proportional to built-up cover).
    """
    if pressure < 0.0:
        raise ValueError("pressure must be non-negative")
    urban = grid.group_fraction(LandCoverGroup.URBAN).astype(np.float64)
    density = np.asarray(
        [_window_mean(urban, parcel, _DENSITY_WINDOW) for parcel in parcels],
        dtype=np.float64,
    )
    return 1.0 / (1.0 + float(pressure) * density)


def _window_mean(plane: np.ndarray, parcel: ParcelIndex, window: int) -> float:
    radius = window // 2
    height, width = plane.shape
    rows = slice(max(parcel.row - radius, 0), min(parcel.row + radius + 1, height))
    cols = slice(max(parcel.col - radius, 0), min(parcel.col + radius + 1, width))
    return float(plane[rows, cols].mean())


def greedy_choice(
    utility: np.ndarray, parcels: Sequence[ParcelIndex], count: int
) -> tuple[ParcelIndex, ...]:
    """The myopic bottom-up layer: take the ``count`` most useful cells."""
    values = np.asarray(utility, dtype=np.float64).reshape(-1)
    if values.size != len(parcels):
        raise ValueError(f"got {values.size} utilities for {len(parcels)} parcels")
    if count < 0:
        raise ValueError("count must be non-negative")
    order = sorted(
        range(len(parcels)), key=lambda index: (-values[index], parcels[index])
    )
    return tuple(parcels[index] for index in order[: min(count, len(parcels))])


def compromise_scores(
    criteria: np.ndarray, weights: np.ndarray, order: float = 2.0
) -> np.ndarray:
    """Cohon's ``L_p`` distance to the ideal point; **lower is better**.

    ``L_p(c) = [ sum_m (w_m * (f*_m - f_m(c)) / (f*_m - f^-_m))^p ]^(1/p)``
    with the ideal ``f*`` and anti-ideal ``f^-`` taken over the alternatives
    currently on the table.  A criterion on which every alternative scores the
    same has no spread and therefore contributes nothing.
    """
    if order <= 0.0:
        raise ValueError("order must be positive")
    values = np.asarray(criteria, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("criteria must be a (alternatives, criteria) matrix")
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weight.size != values.shape[1]:
        raise ValueError(
            f"got {weight.size} weights for {values.shape[1]} criteria columns"
        )
    if values.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    ideal = values.max(axis=0)
    anti_ideal = values.min(axis=0)
    spread = ideal - anti_ideal
    regret = np.divide(
        ideal - values,
        spread,
        out=np.zeros_like(values),
        where=spread > 0.0,
    )
    return np.power(np.power(weight * regret, order).sum(axis=1), 1.0 / order)


def compromise_allocation(
    parcels: Sequence[ParcelIndex],
    criteria: np.ndarray,
    weights: np.ndarray,
    budget: int | None,
    order: float = 2.0,
) -> tuple[ParcelIndex, ...]:
    """MOLA-style allocation of the top-``budget`` alternatives.

    Design note: MitM has no conversion budget of its own, so the note's
    recommendation is followed - the budget is enforced *inside* the MODM step
    by keeping the best ``B`` by compromise score.

    Ties are broken by parcel order, which matters more than it looks: with few
    alternatives the normalised regret saturates at 0 and 1, so equal-weight
    compromise scores tie often.  A deterministic rule keeps seeded runs
    reproducible.
    """
    scores = compromise_scores(criteria, weights, order=order)
    if scores.size != len(parcels):
        raise ValueError(f"got {scores.size} scores for {len(parcels)} parcels")
    ranked = sorted(
        range(len(parcels)), key=lambda index: (scores[index], parcels[index])
    )
    if budget is not None:
        ranked = ranked[: max(int(budget), 0)]
    return tuple(parcels[index] for index in ranked)
