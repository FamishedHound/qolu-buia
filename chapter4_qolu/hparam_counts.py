"""The headline counts: how many perturbations left the method ranking alone.

Split out of :mod:`chapter4_qolu.hparam_summary`, which assembles the payload,
so that *counting* stays separate from *building* - the same split
:mod:`chapter4_qolu.hparam_catalogue` makes out of
:mod:`chapter4_qolu.hparam_sensitivity`.  Everything here is pure: it takes the
per-perturbation blocks the summary has already built and returns the
``overall`` section of the payload.

Design note (why every count is reported twice).  ``preserved / total`` spans
every *effective* perturbation - every one that was actually run - which is
what a reader who has just been told "21 perturbations were run" expects to
see.  ``informative_preserved / n_informative`` drops the ones whose runs came
back identical to the baseline's (:mod:`chapter4_qolu.hparam_inert`), and it is
that pair the report quotes, because a perturbation that changed nothing cannot
have preserved anything.  Both are in ``hparam_summary.json``, so a reader can
check one against the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chapter4_qolu.hparam_inert import INERT_REASON, informative_perturbations
from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS

__all__ = ["RANKING_VIEWS", "overall_counts"]

#: Design note (which views are counted).  The campaign carries three readings
#: of the same metrics.  ``minmax`` is an increasing affine map of ``raw``
#: within a cell, so it *cannot* order the methods differently and is not
#: counted separately; ``absolute`` clips distances at a physical saturation
#: distance before averaging and genuinely can, so it is.
RANKING_VIEWS: tuple[str, ...] = ("raw", "absolute")


def _mean(values: Sequence[float | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _verdicts(
    block: Mapping[str, Any], metric: str, view: str
) -> list[Mapping[str, Any]]:
    """The per-scenario ranking verdicts of one block, for one metric and view."""
    return [
        (
            scenario["metrics"][metric]
            if view == "raw"
            else scenario["metrics"][metric][view]
        )
        for scenario in block["scenarios"]
    ]


def _preserved(blocks: Sequence[Mapping[str, Any]], metric: str, view: str) -> int:
    """Blocks whose ranking held in *every* scenario they were run in."""
    count = 0
    for block in blocks:
        verdicts = _verdicts(block, metric, view)
        if verdicts and all(verdict["ranking_preserved"] for verdict in verdicts):
            count += 1
    return count


def _all_metrics_preserved(blocks: Sequence[Mapping[str, Any]]) -> int:
    """Blocks whose ranking held on every metric in every scenario."""
    return sum(
        1
        for block in blocks
        if all(
            scenario["metrics"][metric]["ranking_preserved"]
            for scenario in block["scenarios"]
            for metric in PUBLISHED_METRICS
        )
    )


def _view_counts(
    effective: Sequence[Mapping[str, Any]],
    informative: Sequence[Mapping[str, Any]],
    metric: str,
    view: str,
) -> dict[str, Any]:
    """One metric x view cell of the headline table."""
    cells = [
        verdict for block in effective for verdict in _verdicts(block, metric, view)
    ]
    preserved = _preserved(effective, metric, view)
    return {
        "preserved": preserved,
        "total": len(effective),
        "share": (preserved / len(effective)) if effective else None,
        "informative_preserved": _preserved(informative, metric, view),
        "n_informative": len(informative),
        "cells_preserved": sum(1 for cell in cells if cell["ranking_preserved"]),
        "n_cells": len(cells),
        "mean_kendall_tau": _mean([cell["kendall_tau"] for cell in cells]),
        "mean_kendall_tau_informative": _mean(
            [
                verdict["kendall_tau"]
                for block in informative
                for verdict in _verdicts(block, metric, view)
            ]
        ),
    }


def overall_counts(blocks: Sequence[Mapping[str, Any]], n_no_op: int) -> dict[str, Any]:
    """How many perturbations preserved the ranking, per metric and per view.

    A perturbation counts as preserving a metric's ranking only when it does so
    in **every** scenario it was run in; the per-cell counts are reported beside
    that, because a driver run over all four scenarios has four times as many
    chances to break.
    """
    effective = [block for block in blocks if not block["no_op"] and block["scenarios"]]
    informative = informative_perturbations(effective)
    metrics: dict[str, Any] = {}
    for metric in PUBLISHED_METRICS:
        entry: dict[str, Any] = {}
        for view in RANKING_VIEWS:
            values = _view_counts(effective, informative, metric, view)
            if view == "raw":
                entry.update(values)
            else:
                entry[view] = values
        metrics[metric] = entry
    return {
        "n_perturbations": len(blocks),
        "n_effective": len(effective),
        "n_no_op": n_no_op,
        "n_inert": len(effective) - len(informative),
        "n_informative": len(informative),
        "inert_labels": [
            str(block.get("label", "")) for block in effective if block.get("inert")
        ],
        "inert_reason": INERT_REASON,
        "n_scenarios": max((len(b["scenarios"]) for b in effective), default=0),
        "metrics": metrics,
        "all_metrics_preserved": _all_metrics_preserved(effective),
        "all_metrics_preserved_informative": _all_metrics_preserved(informative),
    }
