"""``per_seed.csv``: one full-precision row per campaign run.

``tables.md`` rounds - it has to, it is a table - and ``summary.json`` is
nested.  This file carries the underlying numbers at full precision in a form
a spreadsheet, R or pandas can open without any knowledge of this package.

Every metric is written with :func:`repr`, i.e. the shortest decimal string
that round-trips to the same IEEE double, so nothing is lost and nothing
invented: a value of ``0.123456789`` is written ``0.123456789``, not
``0.12`` and not ``0.12345678899999999``.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from chapter4_qolu.reporting.aggregate import ALL_METRICS
from chapter4_qolu.reporting.derived import with_derived_metrics

__all__ = ["PER_SEED_COLUMNS", "per_seed_rows", "render_per_seed_csv"]

#: Column order of ``per_seed.csv``.  ``window`` is the scan's window label
#: (empty for a campaign that did not pin one).
PER_SEED_COLUMNS: tuple[str, ...] = (
    "configuration",
    "scenario",
    "method",
    "seed",
    "window",
    *ALL_METRICS,
    *(f"absolute_{metric}" for metric in ALL_METRICS),
    "total_conversions",
    "saturation_iteration",
    "runtime_seconds",
)


def _number(value: Any) -> str:
    """Full-precision decimal text, or empty for a missing value."""
    if value is None:
        return ""
    if isinstance(value, bool):  # pragma: no cover - defensive
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def _window_label(record: Mapping[str, Any]) -> str:
    window = record.get("window") or {}
    label = window.get("label") if isinstance(window, Mapping) else None
    if label:
        return str(label)
    origin = window.get("raster_origin") if isinstance(window, Mapping) else None
    if origin:
        return f"row{origin[0]}_col{origin[1]}"
    return ""


def per_seed_rows(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """One row per run record, sorted by configuration / scenario / method / seed."""
    rows: list[dict[str, str]] = []
    for original in records:
        # A record that lacks c_align is scored on it from its stored grids.
        record = with_derived_metrics(original)
        raw = record.get("final_metrics") or {}
        absolute = record.get("final_metrics_absolute") or {}
        configuration = record.get("configuration") or {}
        row = {
            "configuration": str(configuration.get("label", "")),
            "scenario": str(record.get("scenario", "")),
            "method": str(record.get("method", "")),
            "seed": str(record.get("seed", "")),
            "window": _window_label(record),
            "total_conversions": _number(record.get("total_conversions")),
            "saturation_iteration": _number(record.get("saturation_iteration")),
            "runtime_seconds": _number(record.get("runtime_seconds")),
        }
        for metric in ALL_METRICS:
            row[metric] = _number(raw.get(metric))
            row[f"absolute_{metric}"] = _number(absolute.get(metric))
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["configuration"],
            row["scenario"],
            row["method"],
            row["seed"],
        )
    )
    return rows


def render_per_seed_csv(records: Sequence[Mapping[str, Any]]) -> str:
    """The whole ``per_seed.csv`` document."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(PER_SEED_COLUMNS), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(per_seed_rows(records))
    return buffer.getvalue()
