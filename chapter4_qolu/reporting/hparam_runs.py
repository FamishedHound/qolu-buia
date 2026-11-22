"""``hparam_runs.csv``: one full-precision row per perturbed run.

The markdown report rounds - it has to, it is a table - and the summary is
nested.  This file carries every run behind the analysis at full precision, in
a form a spreadsheet, R or pandas can open without any knowledge of this
package, with the perturbation that produced it in the leading columns.

The metric columns are :data:`chapter4_qolu.reporting.per_seed.PER_SEED_COLUMNS`
verbatim, produced by the campaign's own row builder, so a row here and the
corresponding row of an ordinary ``per_seed.csv`` agree byte for byte.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from typing import Any

from chapter4_qolu.reporting.per_seed import PER_SEED_COLUMNS, per_seed_rows

__all__ = [
    "BASELINE_LABEL",
    "HPARAM_RUN_COLUMNS",
    "PERTURBATION_COLUMNS",
    "hparam_run_rows",
    "render_hparam_runs_csv",
]

#: Columns added in front of the ordinary per-seed ones, so a spreadsheet can
#: group the runs by what was perturbed.  A baseline run leaves every one of
#: them empty except the first, which reads ``baseline``.
PERTURBATION_COLUMNS: tuple[str, ...] = (
    "perturbation",
    "hyperparameter",
    "direction",
    "baseline_value",
    "applied_value",
    "requested_value",
    "rounded",
    "clamped",
)
HPARAM_RUN_COLUMNS: tuple[str, ...] = (*PERTURBATION_COLUMNS, *PER_SEED_COLUMNS)

#: What a run with no perturbation is called in the first column.
BASELINE_LABEL = "baseline"


def _exact(value: Any) -> str:
    """Full-precision decimal text, or empty for a missing value."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def hparam_run_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """One row per run record, baseline first, then perturbation by perturbation."""
    rows: list[dict[str, str]] = []
    for record in records:
        # The campaign's own row builder, one record at a time, so the metric
        # columns cannot drift from per_seed.csv's.
        row = dict(per_seed_rows([record])[0])
        item = record.get("hyperparameter") or {}
        row.update(
            {
                "perturbation": str(item.get("label") or BASELINE_LABEL),
                "hyperparameter": str(item.get("hyperparameter") or ""),
                "direction": str(item.get("direction") or ""),
                "baseline_value": _exact(item.get("baseline")),
                "applied_value": _exact(item.get("value")),
                "requested_value": _exact(item.get("requested")),
                "rounded": _exact(item.get("rounded")),
                "clamped": _exact(item.get("clamped")),
            }
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["perturbation"] != BASELINE_LABEL,
            row["perturbation"],
            row["scenario"],
            row["method"],
            row["seed"],
        )
    )
    return rows


def render_hparam_runs_csv(records: Sequence[Mapping[str, Any]]) -> str:
    """The whole ``hparam_runs.csv`` document."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(HPARAM_RUN_COLUMNS), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(hparam_run_rows(records))
    return buffer.getvalue()
