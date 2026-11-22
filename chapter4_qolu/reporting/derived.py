"""Metrics a run record can be scored on after the fact, from its own grids.

``c_align``, the total-variation conversion metric of
:func:`chapter4_qolu.env.evaluation.evaluate_landscape`, is a function of the
initial and final dominant-class maps alone, which a run record stores, so a
record that does not carry the metric is scored from them.  That is exact
rather than approximate: the stored maps *are* the
``ParcelGrid.dominant_class()`` arrays the metric reads during a run
(``tests/test_derived_metrics.py`` checks the two routes agree on a real run).

Nothing is estimated: a record without stored grids stays without the metric,
and the aggregation counts, per metric, how many values were recorded during a
run, how many were scored from stored grids and how many are missing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from chapter4_qolu.env.class_grids import decode_class_grid
from chapter4_qolu.env.metrics import c_align_from_classes

__all__ = [
    "DERIVED_METRICS",
    "FROM_STORED_GRIDS",
    "LCM2015_DOI",
    "MISSING",
    "RECORDED",
    "MissingLandCoverGrids",
    "derived_metric_counts",
    "has_stored_grids",
    "metric_source",
    "missing_grids_message",
    "with_derived_metrics",
]

#: Metrics that can be recomputed from a record's stored grids.
DERIVED_METRICS: tuple[str, ...] = ("c_align",)

#: Where a record's value of a derived metric came from.
RECORDED = "recorded"
FROM_STORED_GRIDS = "from_stored_grids"
MISSING = "missing"

#: The land-cover product the Chapter 4 records were computed on.
LCM2015_DOI = "https://doi.org/10.5285/711c8dc1-0f4e-42ad-a703-8b5d19c92247"

_VIEWS: tuple[str, ...] = ("final_metrics", "final_metrics_absolute")


def _stored_grids(record: Mapping[str, Any]) -> tuple[Any, Any] | None:
    initial, final = record.get("initial_grid"), record.get("final_grid")
    if not isinstance(initial, Mapping) or not isinstance(final, Mapping):
        return None
    return initial, final


def has_stored_grids(record: Mapping[str, Any]) -> bool:
    """Whether a record carries both of its land-cover grids."""
    return _stored_grids(record) is not None


class MissingLandCoverGrids(RuntimeError):
    """A command needs land-cover grids that the run records do not carry."""


def missing_grids_message(what: str, where: object, n_records: int) -> str:
    """Why ``what`` cannot be done without the grids, and how to get them back.

    Design note (published records): the run records distributed with this
    package leave ``initial_grid`` and ``final_grid`` out.  They are the
    per-parcel land-cover classes of the study region, derived from the UKCEH
    Land Cover Map 2015, whose licence does not allow the data to be
    redistributed, so a reader regenerates them by running the experiments on
    their own copy of the product.
    """
    return (
        f"{what} needs the land-cover grids (initial_grid, final_grid) of the "
        f"run records, and {n_records} record(s) under {where} carry none.  "
        "The records published with this package leave the grids out because "
        "they are derived from the UKCEH Land Cover Map 2015, whose licence "
        "does not allow them to be redistributed.  To regenerate them, "
        "download LCM2015 (1km dominant aggregate class, GB) from UKCEH "
        f"({LCM2015_DOI}), place it as README.md describes ('Obtaining the "
        "raster used in Chapter 4') and rerun the experiments (step 1 of "
        "'Reproducing the Chapter 4 tables', without --resume or under a new "
        "--tag), which writes the records again with their grids."
    )


def metric_source(record: Mapping[str, Any], metric: str = "c_align") -> str:
    """``recorded``, ``from_stored_grids`` or ``missing`` for one record."""
    if metric in (record.get("final_metrics") or {}):
        derived = (record.get("derived_metrics") or {}).get(metric)
        return derived if derived == FROM_STORED_GRIDS else RECORDED
    return FROM_STORED_GRIDS if _stored_grids(record) is not None else MISSING


def with_derived_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``record`` whose metric views carry every derivable metric.

    The input is never modified.  A metric the run already recorded is kept as
    it is; one it did not is scored from the stored grids and named in the
    copy's ``derived_metrics`` block.  ``c_align`` has no distance term, so the
    RAW and absolute views carry the same value, as they do for ``woodland``.
    """
    out = dict(record)
    raw = record.get("final_metrics")
    if not isinstance(raw, Mapping) or "c_align" in raw:
        return out
    grids = _stored_grids(record)
    if grids is None:
        return out
    value = c_align_from_classes(
        decode_class_grid(grids[0]), decode_class_grid(grids[1])
    )
    for view in _VIEWS:
        block = record.get(view)
        if isinstance(block, Mapping):
            out[view] = {**block, "c_align": value}
    out["derived_metrics"] = {
        **dict(record.get("derived_metrics") or {}),
        "c_align": FROM_STORED_GRIDS,
    }
    return out


def derived_metric_counts(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    """Per derived metric: how many records recorded it, derived it, lack it."""
    rows = list(records)
    return {
        metric: {
            source: sum(1 for record in rows if metric_source(record, metric) == source)
            for source in (RECORDED, FROM_STORED_GRIDS, MISSING)
        }
        for metric in DERIVED_METRICS
    }
