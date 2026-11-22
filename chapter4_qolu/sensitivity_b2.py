"""Stage B2: does the winning *recipe* work on any other window?

Stage A selects the recipe - cap, QOLU reward reading, baseline variants - on
three environments, and stage B confirms it over four scenarios and three
seeds.  Neither separates the **recipe** from the **window**, and the window
can dominate.  On a window that holds essentially no heath or bog, *every*
developed parcel is "far" from the habitat, and on one that is almost
entirely built up already, everything new is next to existing urban land, so
the Land Use and Urban metrics cannot separate four methods by more than about
0.01.  The published cells are separated by 0.05-0.10.  A configuration can
therefore reproduce the published *ordering* on such a window while being
nowhere near the published *values*.

Stage B2 runs the winning recipe on **all** retained windows plus the whole
region, at Scenario 1 and seed 0, and scores each one lexicographically:

1. tie-aware strict first places; **3 of 3 is required to qualify** - a window
   that does not reproduce the published ordering is not a candidate however
   well its numbers line up;
2. Kendall's tau-a on the values;
3. the MAE against the 12 published Scenario-1 cells, after fitting
   ``max_distance`` per window offline (:mod:`chapter4_qolu.rescore`) - the
   constant is unknown, so each window is given its own best case;
4. a **separation** statistic: the mean absolute gap between adjacent methods,
   averaged over the three metrics.  The published tables separate adjacent
   methods by 0.05-0.10 (mean exactly 1/15); a window whose methods differ by
   0.005 is not reproducing that pattern whatever its ordering says.

Design note (separation tolerance): "within a factor of two" of the published
mean gap, i.e. a ratio in [0.5, 2].  It is a coarse test, and it is meant to
be: the point is to reject windows where the metrics are degenerate, not to
tune a gap.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chapter4_qolu.methods import PUBLISHED_METHODS, canonical_method
from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS
from chapter4_qolu.reporting.published import published_means, value_closeness
from chapter4_qolu.rescore import CANDIDATE_DISTANCES_KM, rescore_record
from chapter4_qolu.scan_scoring import WindowScore, score_window
from chapter4_qolu.window_scan import (
    MAX_SEA_FRACTION,
    SOUTH_EAST_ENGLAND,
    WINDOW_SIZE_KM,
    WINDOW_STRIDE_KM,
    ScanWindow,
    candidate_windows,
    region_window,
    retain_windows,
)

__all__ = [
    "PUBLISHED_SEPARATION",
    "SEPARATION_TOLERANCE",
    "B2Score",
    "b2_environments",
    "fits_raster",
    "retained_windows",
    "score_b2_window",
    "separation",
]

#: Sort key stand-in for "not measured", so a missing number never wins.
_WORST = float("-inf")
#: Design note: a separation ratio inside [1/2, 2] counts as "the same order of
#: magnitude as the published one"; see the module docstring.
SEPARATION_TOLERANCE: float = 2.0


def separation(values: Mapping[str, Mapping[str, float]]) -> float | None:
    """Mean absolute gap between adjacent methods, averaged over the metrics.

    Four methods give three adjacent gaps per metric; the statistic is the
    mean of all of them.  ``None`` when no metric had at least two methods.
    """
    gaps: list[float] = []
    for metric in PUBLISHED_METRICS:
        per_method = values.get(metric) or {}
        ordered = sorted((float(value) for value in per_method.values()), reverse=True)
        gaps.extend(
            abs(first - second) for first, second in itertools.pairwise(ordered)
        )
    if not gaps:
        return None
    return float(np.mean(gaps))


def _published_separation() -> float:
    return float(separation(published_means(1)) or 0.0)


#: The published Table-4 separation: every adjacent pair differs by 0.05 or
#: 0.10 in all three columns, so the mean gap is exactly 1/15.
PUBLISHED_SEPARATION: float = _published_separation()


def fits_raster(raster: Any, window: ScanWindow) -> bool:
    """Can this raster supply the whole window?

    The scan grid is defined in British National Grid kilometres, not in
    raster rows, so a raster that does not cover the whole Southeast box - a
    small test fixture, or a differently-cropped product - simply has fewer
    candidate windows.  That is a property of the data, not an error, so it is
    filtered rather than raised.
    """
    from chapter4_qolu.data.lcm_loader import bng_origin_to_index

    try:
        row, col = bng_origin_to_index(raster, *window.bng_top_left_km)
    except ValueError:
        return False
    rows, cols = window.shape
    return (
        row >= 0
        and col >= 0
        and row + rows <= raster.height
        and col + cols <= raster.width
    )


def retained_windows(
    raster_path: Path | str,
    box: Any = SOUTH_EAST_ENGLAND,
    size_km: int = WINDOW_SIZE_KM,
    stride_km: int = WINDOW_STRIDE_KM,
    max_sea_fraction: float = MAX_SEA_FRACTION,
) -> tuple[ScanWindow, ...]:
    """The scan grid's windows that fit the raster and survive the sea filter."""
    from chapter4_qolu.data.lcm_loader import load_lcm_raster

    raster = load_lcm_raster(raster_path)
    candidates = [
        window
        for window in candidate_windows(box, size_km, stride_km)
        if fits_raster(raster, window)
    ]
    if not candidates:
        return ()
    kept, _ = retain_windows(raster, candidates, max_sea_fraction)
    return tuple(item.window for item in kept)


def b2_environments(
    raster_path: Path | str, include_region: bool = True
) -> tuple[ScanWindow, ...]:
    """Every retained scan window, plus the whole-region box when it fits."""
    from chapter4_qolu.data.lcm_loader import load_lcm_raster

    windows = retained_windows(raster_path)
    if not include_region:
        return windows
    region = region_window()
    if not fits_raster(load_lcm_raster(raster_path), region):
        return windows
    return (*windows, region)


def _values(
    records: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    max_distance_km: float | None,
) -> dict[str, dict[str, float]]:
    """``values[metric][published method]`` re-scored at one saturation distance."""
    wanted = set(methods)
    values: dict[str, dict[str, float]] = {}
    for record in records:
        method = str(record.get("method", ""))
        if method not in wanted:
            continue
        metrics = rescore_record(record, max_distance_km)
        if metrics is None:
            metrics = record.get("final_metrics_absolute") or {}
        for metric in PUBLISHED_METRICS:
            if metric in metrics:
                values.setdefault(metric, {})[canonical_method(method)] = float(
                    metrics[metric]
                )
    return values


def _raw_values(
    records: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> dict[str, dict[str, float]]:
    wanted = set(methods)
    values: dict[str, dict[str, float]] = {}
    for record in records:
        method = str(record.get("method", ""))
        if method not in wanted:
            continue
        metrics = record.get("final_metrics") or {}
        for metric in PUBLISHED_METRICS:
            if metric in metrics:
                values.setdefault(metric, {})[canonical_method(method)] = float(
                    metrics[metric]
                )
    return values


@dataclass(frozen=True)
class B2Score:
    """How one window does under the winning recipe."""

    window: ScanWindow
    window_score: WindowScore
    #: The saturation distance that minimised the MAE on this window.
    max_distance_km: float | None
    value_mae: float | None
    #: Mean adjacent-method gap at that distance.
    separation: float | None
    n_records: int

    @property
    def qualifies(self) -> bool:
        """Did it reproduce the published ordering on all three metrics?"""
        return self.window_score.first_place_count == len(PUBLISHED_METRICS)

    @property
    def separation_ratio(self) -> float | None:
        if self.separation is None or PUBLISHED_SEPARATION <= 0.0:
            return None
        return float(self.separation / PUBLISHED_SEPARATION)

    @property
    def separation_in_band(self) -> bool:
        ratio = self.separation_ratio
        if ratio is None:
            return False
        return 1.0 / SEPARATION_TOLERANCE <= ratio <= SEPARATION_TOLERANCE

    @property
    def sort_key(self) -> tuple[int, float, float, float]:
        """First places, then tau, then closeness, then separation plausibility.

        Closeness and separation both enter negated, because for both a
        *smaller* number is better: a smaller MAE is nearer the published
        cells, and a separation ratio nearer 1 is nearer the published spread.
        """
        closeness = _WORST if self.value_mae is None else -float(self.value_mae)
        ratio = self.separation_ratio
        plausibility = _WORST if ratio is None or ratio <= 0.0 else -abs(np.log(ratio))
        return (
            self.window_score.first_place_count,
            self.window_score.kendall_sum,
            closeness,
            float(plausibility),
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "window": self.window.as_record(),
            "score": self.window_score.as_record(),
            "max_distance_km": self.max_distance_km,
            "value_mae": self.value_mae,
            "separation": self.separation,
            "separation_ratio": self.separation_ratio,
            "separation_in_band": self.separation_in_band,
            "published_separation": PUBLISHED_SEPARATION,
            "qualifies": self.qualifies,
            "n_records": self.n_records,
        }


def score_b2_window(
    records: Sequence[Mapping[str, Any]],
    window: ScanWindow,
    methods: Sequence[str] = PUBLISHED_METHODS,
    distances: Sequence[float | None] = CANDIDATE_DISTANCES_KM,
) -> B2Score:
    """Score one window: ordering from the RAW view, values from the best fit.

    The ordering is taken from the RAW metrics, exactly as stage A and the
    window scan take it, so the two are comparable.  The closeness and the
    separation are taken at the ``max_distance`` that suits *this* window best,
    because the paper's constant is unknown and giving each window its own best
    case is the conservative choice when the question is "could this window
    have produced those numbers at all".
    """
    present = {
        str(record.get("method", ""))
        for record in records
        if str(record.get("method", "")) in set(methods)
    }
    ordering = score_b2_ordering(records, methods)
    best_distance: float | None = None
    best_mae: float | None = None
    best_separation: float | None = None
    for distance in distances:
        values = _values(records, methods, distance)
        mae = value_closeness(values, scenario=1)
        if mae is None:
            continue
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_distance = distance
            best_separation = separation(values)
    return B2Score(
        window=window,
        window_score=ordering,
        max_distance_km=best_distance,
        value_mae=best_mae,
        separation=best_separation,
        n_records=len(present),
    )


def score_b2_ordering(
    records: Sequence[Mapping[str, Any]], methods: Sequence[str] = PUBLISHED_METHODS
) -> WindowScore:
    """The tie-aware ordering score of one window, from the RAW view."""
    return score_window(_raw_values(records, methods))
