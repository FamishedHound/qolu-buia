"""Scoring one cell of the sensitivity sweep, and counting a 12-cell match.

A *cell* is one (window, configuration) pair: four runs, one per method, all at
Scenario 1 and seed 0.  It is scored exactly as the window scan scores a window
- strict first places, then Kendall's tau-a on the values, then QOLU + BUIA's
mean min-max - with one statistic added:

**value closeness** - the mean absolute difference between the cell's values
and the published Table-4 cells, on the view whose scale the published
numbers actually have (see :mod:`chapter4_qolu.reporting.published`).  It is a
*tie-break between configurations*, never a claim that the published values can
be reproduced.

The 12-cell stage-B count is reported **twice**, side by side:

alphabetical
    the campaign's own reading, from ``summary.json``'s ``ranking`` - which
    ``reporting.aggregate.rank_methods`` produces by sorting on the value and
    breaking ties by method *name*.  Reproducible, and what ``pattern_check.md``
    already reports, but it silently hands every tie to ``ltos`` (first
    alphabetically) and counts it against ``qolu_buia`` (last).

tie-aware
    computed from the values, the way ``scan_scoring`` does: on an emphasised
    metric QOLU + BUIA has to be *strictly* ahead of all three baselines, away
    from one it only has to be within a tie of the best, and the baselines have
    to be ordered P-MADDPG >= LToS >= MitM rather than strictly.

Design note (tie-aware baseline order): a tie between two baselines counts
neither for nor against the published order, matching ``kendall_tau_on_values``
in :mod:`chapter4_qolu.scan_scoring`.  The published tables themselves show
ties (QOLU + BUIA 0.75 = P-MADDPG 0.75 for Woodland in Scenario 2), so a
reading that treated every tie as a failure would reject the published tables.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chapter4_qolu.methods import PUBLISHED_METHODS, canonical_method
from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS, REFERENCE_METHOD
from chapter4_qolu.reporting.pattern import EMPHASISED_METRICS, PUBLISHED_RANKING
from chapter4_qolu.reporting.published import PUBLISHED_CLOSENESS_VIEW, value_closeness
from chapter4_qolu.scan_scoring import WindowScore, score_window
from chapter4_qolu.sensitivity_plan import SensitivityConfiguration
from chapter4_qolu.window_scan import ScanWindow

__all__ = [
    "ConfigurationScore",
    "best_per_configuration",
    "count_pattern_matches_tie_aware",
    "score_configuration",
]

#: Values this close together count as identical - the same tolerance
#: ``scan_scoring`` uses, and for the same reason.
_TOLERANCE: float = 1e-12
#: Sort key stand-in for "this configuration has no closeness measurement at
#: all", so a cell whose runs are missing can never outrank a measured one.
_NO_CLOSENESS: float = float("-inf")


def _metric_values(
    records: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    key: str,
) -> dict[str, dict[str, float]]:
    """``values[metric][published method]`` from the chosen runs of a cell.

    Only the methods the configuration actually selected are used, and each is
    filed under the *published* method it stands in for, so a variant is
    compared against its parent's published row.
    """
    wanted = set(methods)
    values: dict[str, dict[str, float]] = {}
    for record in records:
        method = str(record.get("method", ""))
        if method not in wanted:
            continue
        metrics = record.get(key) or {}
        for metric in PUBLISHED_METRICS:
            if metric in metrics:
                values.setdefault(metric, {})[canonical_method(method)] = float(
                    metrics[metric]
                )
    return values


@dataclass(frozen=True)
class ConfigurationScore:
    """How one (window, configuration) cell compares with the published Table 4."""

    configuration: SensitivityConfiguration
    window: ScanWindow
    window_score: WindowScore
    #: Mean absolute difference from Table 4 on the published-scale view.
    value_mae: float | None
    #: The same statistic on the cross-method min-max view, for comparison.
    value_mae_minmax: float | None
    n_records: int

    @classmethod
    def from_parts(
        cls,
        configuration: SensitivityConfiguration,
        window: ScanWindow,
        first: int,
        tau: float,
        minmax: float,
        mae: float | None,
        mae_minmax: float | None,
        n: int,
    ) -> ConfigurationScore:
        """Build a score directly from its numbers (tests and re-ranking)."""
        return cls(
            configuration=configuration,
            window=window,
            window_score=WindowScore(first, tau, minmax, ()),
            value_mae=mae,
            value_mae_minmax=mae_minmax,
            n_records=n,
        )

    @property
    def sort_key(self) -> tuple[int, float, float, float]:
        """Lexicographic key: first places, then tau, then closeness, then margin.

        Closeness enters **negated**, because a smaller absolute difference is
        a better one, and a missing measurement sorts below every measured one.
        """
        closeness = _NO_CLOSENESS if self.value_mae is None else -float(self.value_mae)
        return (
            self.window_score.first_place_count,
            self.window_score.kendall_sum,
            closeness,
            self.window_score.reference_minmax_mean,
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration.as_record(),
            "window": self.window.as_record(),
            "score": self.window_score.as_record(),
            "value_mae": self.value_mae,
            "value_mae_minmax": self.value_mae_minmax,
            "closeness_view": PUBLISHED_CLOSENESS_VIEW,
            "n_records": self.n_records,
        }


def _minmax_values(
    values: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    from chapter4_qolu.reporting.aggregate import min_max_normalise

    return {metric: min_max_normalise(cell) for metric, cell in values.items()}


def score_configuration(
    records: Sequence[Mapping[str, Any]],
    configuration: SensitivityConfiguration,
    window: ScanWindow,
    scenario: int = 1,
) -> ConfigurationScore:
    """Score one cell from the stage-A records of its window."""
    methods = configuration.methods
    raw = _metric_values(records, methods, "final_metrics")
    absolute = _metric_values(records, methods, "final_metrics_absolute")
    chosen = {
        str(record.get("method", ""))
        for record in records
        if str(record.get("method", "")) in set(methods)
    }
    closeness_values = absolute if PUBLISHED_CLOSENESS_VIEW == "absolute" else raw
    return ConfigurationScore(
        configuration=configuration,
        window=window,
        window_score=score_window(raw),
        value_mae=value_closeness(closeness_values, scenario=scenario),
        value_mae_minmax=value_closeness(_minmax_values(raw), scenario=scenario),
        n_records=len(chosen),
    )


def best_per_configuration(
    scores: Sequence[ConfigurationScore],
) -> list[ConfigurationScore]:
    """Each configuration's best window, configurations best first.

    Design note (which window a configuration is judged on): its **best** one.
    The sweep's question is "can any reading of the ambiguities reproduce the
    published ordering", so a configuration that works on one of the three
    pinned windows is a positive answer even if it fails on the other two -
    and the window it needed is reported next to it, never hidden.
    """
    best: dict[str, ConfigurationScore] = {}
    for score in scores:
        label = score.configuration.label
        current = best.get(label)
        if current is None or score.sort_key > current.sort_key:
            best[label] = score
    return sorted(
        best.values(),
        key=lambda item: (item.sort_key, item.configuration.label),
        reverse=True,
    )


def _cell_values(scenario_block: Mapping[str, Any], metric: str) -> dict[str, float]:
    """``values[method]`` of one (scenario, metric) cell, from the RAW means."""
    values: dict[str, float] = {}
    for method, block in (scenario_block.get("methods") or {}).items():
        entry = ((block or {}).get("raw") or {}).get(metric)
        if entry is not None and entry.get("mean") is not None:
            values[canonical_method(str(method))] = float(entry["mean"])
    return values


def _tie_aware_cell(
    scenario: int, metric: str, values: Mapping[str, float], reference: str
) -> dict[str, Any]:
    """Whether one cell reproduces the published ordering, ties respected."""
    emphasised = metric in EMPHASISED_METRICS.get(int(scenario), ())
    reference_value = values.get(reference)
    if reference_value is None:
        return {
            "scenario": int(scenario),
            "metric": metric,
            "emphasised": emphasised,
            "agrees": False,
            "tied_top": False,
            "verdict": f"{reference} did not run in this cell",
        }
    others = {name: value for name, value in values.items() if name != reference}
    ahead = sum(1 for value in others.values() if value - reference_value > _TOLERANCE)
    level = sum(
        1 for value in others.values() if abs(value - reference_value) <= _TOLERANCE
    )
    position_ok = (ahead == 0 and level == 0) if emphasised else ahead <= 1
    baselines = [
        name for name in PUBLISHED_RANKING if name != reference and name in values
    ]
    baseline_ok = all(
        values[first] - values[second] >= -_TOLERANCE
        for first, second in itertools.pairwise(baselines)
    )
    agrees = position_ok and baseline_ok
    if agrees:
        verdict = "yes"
    elif not position_ok:
        expectation = "strictly first" if emphasised else "first or joint first"
        verdict = f"no - {reference} is not {expectation} ({ahead} method(s) ahead)"
    else:
        verdict = "no - the baselines are not in the published order"
    return {
        "scenario": int(scenario),
        "metric": metric,
        "emphasised": emphasised,
        "agrees": agrees,
        "tied_top": ahead == 0 and level > 0,
        "n_ahead": ahead,
        "n_level": level,
        "baselines_in_published_order": baseline_ok,
        "verdict": verdict,
    }


def count_pattern_matches_tie_aware(
    summary: Mapping[str, Any],
    reference: str = REFERENCE_METHOD,
    methods: Sequence[str] = PUBLISHED_METHODS,
) -> tuple[int, int, list[dict[str, Any]]]:
    """The 12-cell count computed from the *values*, so ties are visible."""
    del methods  # canonical_method already files variants under their parent
    reference_method = str(summary.get("reference_method", reference))
    matched = 0
    total = 0
    cells: list[dict[str, Any]] = []
    for configuration in summary.get("configurations", []):
        for scenario_block in configuration.get("scenarios", []):
            scenario = int(scenario_block["scenario"])
            for metric in PUBLISHED_METRICS:
                values = _cell_values(scenario_block, metric)
                if not values:
                    continue
                total += 1
                cell = _tie_aware_cell(scenario, metric, values, reference_method)
                matched += int(cell["agrees"])
                cells.append(cell)
    return matched, total, cells
