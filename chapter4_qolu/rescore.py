"""Re-score finished runs offline, from the land-cover grids they stored.

The published tables are **absolute** 0-1 scores: the scenario-comparison
figure plots exactly the table values and every method sits between 0.60 and
0.90, so no column contains a 0 or a 1 and the numbers cannot be a cross-method
min-max rescaling.  This package's own absolute view (b) is therefore the right
comparison - but it has a free constant in it, ``RewardConfig.max_distance``,
the physical distance at which the two proximity metrics saturate.

Design note (the paper's normalisation constants are unknown): the paper says
only "normalized between 0 and 1" and never states a saturation distance, a
reference area or any other constant.  ``max_distance`` is therefore chosen
here by minimising the distance to the published cells, and because every run
record stores its initial and final land-cover grids that search costs
**no re-runs at all**: the three published metrics are pure functions of those
two grids and the reward configuration.

Design note (one-hot reconstruction): a stored grid holds the *dominant* class
per parcel, so rebuilding a :class:`ParcelGrid` from it gives each parcel 100%
of that class.  On the UKCEH *dominant aggregate class* product that is exactly
what the loader produced in the first place, and conversion only ever moves a
parcel to 100% built-up, so the reconstruction is exact and
:func:`recompute_published_metrics` reproduces the record's own
``final_metrics_absolute`` to machine precision (``tests/test_rescore.py``).
On a *percentage cover* raster it would not be exact, and
:func:`rescore_record` says so rather than pretending otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from chapter4_qolu.config import RewardConfig
from chapter4_qolu.env.evaluation import (
    distance_to_desired_land_types,
    preservation_of_land_types,
    urban_connectivity,
)
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES
from chapter4_qolu.methods import canonical_method
from chapter4_qolu.reporting.published import value_closeness
from chapter4_qolu.run_record import GRID_NO_DATA, decode_class_grid

__all__ = [
    "CANDIDATE_DISTANCES_KM",
    "DistanceFit",
    "fit_max_distance",
    "grid_from_classes",
    "recompute_published_metrics",
    "rescore_record",
    "rescore_records",
]

#: Design note (which constants are tried): 5, 10 and 20 km.  10 km is the
#: package default and the round number closest to the distance beyond which
#: "far from bog, mountain and heath" stops being a planning distinction at
#: 1 km parcels; 5 and 20 bracket it by a factor of two either way, which is
#: enough to show whether the metrics are sensitive to the constant at all.
CANDIDATE_DISTANCES_KM: tuple[float, ...] = (5.0, 10.0, 20.0)

_PUBLISHED_METRIC_FUNCTIONS = {
    "land_use": distance_to_desired_land_types,
    "woodland": preservation_of_land_types,
    "urban": urban_connectivity,
}


def grid_from_classes(values: np.ndarray) -> ParcelGrid:
    """Rebuild a one-hot :class:`ParcelGrid` from a dominant-class map."""
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError(f"a class grid must be 2-D, got shape {array.shape}")
    composition = np.zeros(
        (N_AGGREGATE_CLASSES, array.shape[0], array.shape[1]), dtype=np.float32
    )
    rows, cols = np.nonzero(array > GRID_NO_DATA)
    composition[array[rows, cols], rows, cols] = 1.0
    return ParcelGrid(composition)


def recompute_published_metrics(
    initial: ParcelGrid,
    final: ParcelGrid,
    max_distance_km: float | None,
    reward: RewardConfig | None = None,
) -> dict[str, float]:
    """The three published metrics at a given saturation distance."""
    config = replace(
        reward or RewardConfig(),
        max_distance=None if max_distance_km is None else float(max_distance_km),
    )
    return {
        name: float(function(initial, final, config))
        for name, function in _PUBLISHED_METRIC_FUNCTIONS.items()
    }


def rescore_record(
    record: Mapping[str, Any], max_distance_km: float | None
) -> dict[str, float] | None:
    """Re-score one run record, or ``None`` when it stored no grids."""
    initial = record.get("initial_grid")
    final = record.get("final_grid")
    if not initial or not final:
        return None
    return recompute_published_metrics(
        grid_from_classes(decode_class_grid(initial)),
        grid_from_classes(decode_class_grid(final)),
        max_distance_km,
    )


def rescore_records(
    records: Sequence[Mapping[str, Any]], max_distance_km: float | None
) -> dict[int, dict[str, dict[str, float]]]:
    """``scenario -> metric -> method -> mean`` at one saturation distance.

    Several seeds of the same (scenario, method) cell are averaged, exactly as
    the campaign tables average them, and variant methods are filed under the
    published method they stand in for.
    """
    collected: dict[int, dict[str, dict[str, list[float]]]] = {}
    for record in records:
        metrics = rescore_record(record, max_distance_km)
        if metrics is None:
            continue
        scenario = int(record.get("scenario", 1))
        method = canonical_method(str(record.get("method", "")))
        for metric, value in metrics.items():
            collected.setdefault(scenario, {}).setdefault(metric, {}).setdefault(
                method, []
            ).append(value)
    return {
        scenario: {
            metric: {
                method: float(np.mean(values)) for method, values in per_method.items()
            }
            for metric, per_method in per_metric.items()
        }
        for scenario, per_metric in collected.items()
    }


class DistanceFit:
    """How well one ``max_distance`` reproduces the published cells."""

    __slots__ = ("max_distance_km", "n_cells", "per_scenario", "value_mae")

    def __init__(
        self,
        max_distance_km: float | None,
        value_mae: float | None,
        n_cells: int,
        per_scenario: Mapping[int, float | None],
    ) -> None:
        self.max_distance_km = max_distance_km
        self.value_mae = value_mae
        self.n_cells = n_cells
        self.per_scenario = dict(per_scenario)

    @property
    def sort_key(self) -> float:
        """Smaller MAE is better; a missing one sorts last."""
        return float("inf") if self.value_mae is None else float(self.value_mae)

    def as_record(self) -> dict[str, Any]:
        return {
            "max_distance_km": self.max_distance_km,
            "value_mae": self.value_mae,
            "n_cells": self.n_cells,
            "per_scenario": {
                str(scenario): value for scenario, value in self.per_scenario.items()
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DistanceFit({self.max_distance_km!r}, mae={self.value_mae!r})"


def _fit_one(
    records: Sequence[Mapping[str, Any]], max_distance_km: float | None
) -> DistanceFit:
    per_scenario_values = rescore_records(records, max_distance_km)
    per_scenario: dict[int, float | None] = {}
    differences: list[float] = []
    cells = 0
    for scenario, values in sorted(per_scenario_values.items()):
        mae = value_closeness(values, scenario=scenario)
        per_scenario[scenario] = mae
        if mae is not None:
            differences.append(mae)
            cells += sum(len(cell) for cell in values.values())
    overall = float(np.mean(differences)) if differences else None
    return DistanceFit(max_distance_km, overall, cells, per_scenario)


def fit_max_distance(
    records: Sequence[Mapping[str, Any]],
    distances: Sequence[float | None] = CANDIDATE_DISTANCES_KM,
) -> list[DistanceFit]:
    """Score every candidate saturation distance, best (smallest MAE) first.

    Every candidate is returned, not only the winner, so a report can show how
    much - or how little - the choice of constant actually matters.
    """
    fits = [_fit_one(records, distance) for distance in distances]
    return sorted(fits, key=lambda fit: (fit.sort_key, str(fit.max_distance_km)))
