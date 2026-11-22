"""The published Tables 4-7, transcribed once, plus the closeness statistic.

The four result tables of the IJCAI-2025 paper (Tables 4-7, reproduced in
``thesis/overleaf/chapters/chapter4_paper2.tex`` as
``tab:p2-results-uniform`` and ``tab:p2-results-scenario2/3/4``) are 48 cells:
4 scenarios x 4 methods x 3 metrics, each a ``mean +- SE``.  They are
transcribed here verbatim, and ``tests/test_reporting_published.py`` parses the
chapter source and asserts the transcription is exact, so the constant cannot
drift away from the chapter it quotes.

**What normalisation are they?**  The paper says only "normalized between 0 and
1".  This module answers the question the campaign tables raise: *is the
published normalisation our view (c), the cross-method min-max?*  Under view
(c) the worst method in a column scores exactly 0 and the best exactly 1, so
every one of the 12 published columns would have to contain both a 0 and a 1.
:func:`normalisation_check` counts them.  None of the 12 does - the published
values run from 0.60 to 0.90 and never touch either end - so the answer is
**no**, and the value-closeness statistic is therefore computed against the
view whose *scale* does match.

Design note (which view the closeness statistic uses): view (b), the
absolute-normalised reading with ``RewardConfig.max_distance`` at 10 km.  It is
the only one of the three views that is (i) bounded in [0, 1] by construction,
(ii) not forced to place a 0 and a 1 in every column, and (iii) on this land
cover actually lands in the published 0.6-0.9 band.  View (a) RAW squeezes
``land_use`` into roughly 0.03-0.08 because the normaliser is the grid
diagonal, so an MAE against it would measure the normaliser rather than the
method.  The min-max MAE is carried alongside and reported next to it, never
instead of it.

**None of this makes the published values a reproduction target.**  47 of the
48 cells are multiples of 0.05 (the exception is MitM / Woodland in Scenario 2,
0.61) and every standard error is 0.01, 0.02 or 0.03.  The closeness statistic
exists to *rank configurations against each other* - "this cell of the sweep
lands closer to the published numbers than that one" - and for no other
purpose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chapter4_qolu.methods import PUBLISHED_METHODS, canonical_method
from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS

__all__ = [
    "PUBLISHED_CLOSENESS_VIEW",
    "PUBLISHED_METHOD_ORDER",
    "PUBLISHED_SCENARIOS",
    "PUBLISHED_TABLES",
    "PublishedCell",
    "normalisation_check",
    "published_means",
    "value_closeness",
]

#: Row order of the published tables (best first, as the paper's prose claims).
PUBLISHED_METHOD_ORDER: tuple[str, ...] = PUBLISHED_METHODS
PUBLISHED_SCENARIOS: tuple[int, ...] = (1, 2, 3, 4)

#: Design note: the closeness statistic is computed on view (b); see the module
#: docstring for why, and :func:`normalisation_check` for the evidence.
PUBLISHED_CLOSENESS_VIEW = "absolute"


@dataclass(frozen=True)
class PublishedCell:
    """One published ``mean +- SE`` cell."""

    mean: float
    se: float

    def as_record(self) -> dict[str, float]:
        return {"mean": self.mean, "se": self.se}


def _rows(
    values: Mapping[str, tuple[tuple[float, float], ...]],
) -> dict[str, dict[str, PublishedCell]]:
    return {
        method: {
            metric: PublishedCell(mean=mean, se=se)
            for metric, (mean, se) in zip(PUBLISHED_METRICS, cells)
        }
        for method, cells in values.items()
    }


#: Tables 4-7, verbatim: ``scenario -> method -> metric -> PublishedCell``.
#: Columns are Land Use, Woodland, Urban in the published left-to-right order.
PUBLISHED_TABLES: Mapping[int, Mapping[str, Mapping[str, PublishedCell]]] = {
    # Table 4 - "General performance under Scenario 1"
    1: _rows(
        {
            "ltos": ((0.70, 0.01), (0.75, 0.02), (0.80, 0.03)),
            "pmaddpg": ((0.75, 0.02), (0.80, 0.03), (0.85, 0.02)),
            "mitm": ((0.60, 0.02), (0.65, 0.02), (0.70, 0.01)),
            "qolu_buia": ((0.80, 0.03), (0.85, 0.02), (0.90, 0.01)),
        }
    ),
    # Table 5 - "Distance to Land Use's performance under Scenario 2"
    2: _rows(
        {
            "ltos": ((0.75, 0.02), (0.70, 0.02), (0.75, 0.02)),
            "pmaddpg": ((0.80, 0.01), (0.75, 0.02), (0.80, 0.02)),
            "mitm": ((0.65, 0.02), (0.61, 0.01), (0.65, 0.02)),
            "qolu_buia": ((0.85, 0.02), (0.75, 0.02), (0.80, 0.01)),
        }
    ),
    # Table 6 - "Woodland Loss's performance under Scenario 3"
    3: _rows(
        {
            "ltos": ((0.70, 0.01), (0.75, 0.02), (0.70, 0.02)),
            "pmaddpg": ((0.75, 0.02), (0.80, 0.02), (0.75, 0.02)),
            "mitm": ((0.60, 0.02), (0.65, 0.02), (0.60, 0.01)),
            "qolu_buia": ((0.75, 0.02), (0.85, 0.02), (0.75, 0.02)),
        }
    ),
    # Table 7 - "Distance to Urban's performance under Scenario 4"
    4: _rows(
        {
            "ltos": ((0.70, 0.01), (0.70, 0.01), (0.75, 0.02)),
            "pmaddpg": ((0.75, 0.02), (0.75, 0.02), (0.80, 0.02)),
            "mitm": ((0.60, 0.02), (0.60, 0.01), (0.65, 0.02)),
            "qolu_buia": ((0.75, 0.02), (0.75, 0.02), (0.85, 0.02)),
        }
    ),
}


def published_means(scenario: int) -> dict[str, dict[str, float]]:
    """``values[metric][method]`` of one published table, means only."""
    table = PUBLISHED_TABLES[int(scenario)]
    return {
        metric: {
            method: table[method][metric].mean
            for method in PUBLISHED_METHOD_ORDER
            if metric in table[method]
        }
        for metric in PUBLISHED_METRICS
    }


def normalisation_check() -> dict[str, Any]:
    """Is the published normalisation the cross-method min-max (view c)?

    Under view (c) every (scenario, metric) column must contain an exact 0 and
    an exact 1, because that is what the rescaling produces.  This counts how
    many of the 12 published columns do.
    """
    columns = 0
    with_zero_and_one = 0
    lowest: float | None = None
    highest: float | None = None
    for scenario in PUBLISHED_SCENARIOS:
        for metric, per_method in published_means(scenario).items():
            del metric  # only the column's values matter here
            values = list(per_method.values())
            if not values:
                continue
            columns += 1
            column_low, column_high = min(values), max(values)
            lowest = column_low if lowest is None else min(lowest, column_low)
            highest = column_high if highest is None else max(highest, column_high)
            if column_low == 0.0 and column_high == 1.0:
                with_zero_and_one += 1
    return {
        "question": (
            "does every published (scenario, metric) column contain both a 0 "
            "and a 1, as the cross-method min-max view (c) would force?"
        ),
        "n_columns": columns,
        "n_columns_with_zero_and_one": with_zero_and_one,
        "minmax_compatible": columns > 0 and with_zero_and_one == columns,
        "minimum": lowest,
        "maximum": highest,
        "closeness_view": PUBLISHED_CLOSENESS_VIEW,
        "answer": (
            "no - the published cells run from "
            f"{lowest} to {highest} and never reach either end of [0, 1], so "
            "they are not a cross-method min-max rescaling.  The closeness "
            f"statistic is therefore computed on view ({PUBLISHED_CLOSENESS_VIEW})."
        ),
    }


def value_closeness(
    values: Mapping[str, Mapping[str, float]],
    scenario: int,
    methods: Sequence[str] = PUBLISHED_METHOD_ORDER,
) -> float | None:
    """Mean absolute difference from the published cells of one table.

    ``values`` is ``values[metric][method]`` for one scenario.  Method keys may
    be variants (``fixed_ltos``, ``mitm_learned``, ``pmaddpg_masked``); they are
    compared against their published parent's row.  Cells the campaign does not
    have are skipped rather than counted as a perfect or a total miss, so a
    partial cell is never rewarded for being partial.  ``None`` when nothing at
    all could be compared.
    """
    table = PUBLISHED_TABLES.get(int(scenario))
    if not table:
        return None
    wanted = set(methods)
    differences: list[float] = []
    for metric in PUBLISHED_METRICS:
        per_method = values.get(metric) or {}
        for name, value in per_method.items():
            published_method = canonical_method(name)
            if published_method not in wanted:
                continue
            cell = table.get(published_method, {}).get(metric)
            if cell is None:
                continue
            differences.append(abs(float(value) - cell.mean))
    if not differences:
        return None
    return float(sum(differences) / len(differences))
