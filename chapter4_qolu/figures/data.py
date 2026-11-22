"""Loading a campaign for the figure modules - records, summary and grids.

Everything a figure needs is already on disk: the per-run JSON records and the
campaign's ``summary.json``.  Nothing here runs, trains or loads a raster, so a
whole figure set can be rebuilt from a results folder alone.

A record that does not carry grids gets an empty mapping from
:func:`grids_by_method` rather than an error, so the curve and bar figures
still work for it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chapter4_qolu.reporting.aggregate import load_run_records
from chapter4_qolu.reporting.tables import ordered_methods
from chapter4_qolu.run_record import decode_class_grid

__all__ = [
    "LoadedCampaign",
    "grids_by_method",
    "load_campaign",
    "runs_for",
    "scenarios_in",
    "seeds_in",
    "series_by_method",
]


@dataclass(frozen=True)
class LoadedCampaign:
    """One campaign's records and summary, plus where they came from."""

    directory: Path
    records: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]

    @property
    def tag(self) -> str:
        return str(self.summary.get("campaign", {}).get("tag", self.directory.name))

    @property
    def raster_path(self) -> str | None:
        for record in self.records:
            path = (record.get("raster") or {}).get("path")
            if path:
                return str(path)
        return None

    @property
    def methods(self) -> tuple[str, ...]:
        return ordered_methods({str(r.get("method", "")) for r in self.records})


def load_campaign(directory: Path | str) -> LoadedCampaign:
    """Read a campaign's per-run records and its ``summary.json``."""
    path = Path(directory)
    records = tuple(load_run_records(path))
    summary_path = path / "summary.json"
    summary: Mapping[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return LoadedCampaign(directory=path, records=records, summary=summary)


def scenarios_in(records: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    return tuple(sorted({int(record["scenario"]) for record in records}))


def seeds_in(records: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    return tuple(sorted({int(record["seed"]) for record in records}))


def runs_for(
    records: Sequence[Mapping[str, Any]],
    scenario: int | None = None,
    seed: int | None = None,
) -> list[Mapping[str, Any]]:
    """The records of one scenario and/or seed, in method row order."""
    chosen = [
        record
        for record in records
        if (scenario is None or int(record["scenario"]) == int(scenario))
        and (seed is None or int(record["seed"]) == int(seed))
    ]
    order = {
        method: index
        for index, method in enumerate(
            ordered_methods({str(record.get("method", "")) for record in chosen})
        )
    }
    return sorted(
        chosen,
        key=lambda record: (
            order.get(str(record.get("method", "")), 99),
            int(record["seed"]),
        ),
    )


def grids_by_method(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``method -> (initial, final)`` class grids, skipping records without them."""
    grids: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for record in records:
        initial = record.get("initial_grid")
        final = record.get("final_grid")
        if not initial or not final:
            continue
        grids[str(record.get("method", ""))] = (
            decode_class_grid(initial),
            decode_class_grid(final),
        )
    return grids


def series_by_method(
    records: Sequence[Mapping[str, Any]], series: str
) -> dict[str, dict[int, list[float]]]:
    """``method -> seed -> the per-iteration series``, for the curve figures."""
    collected: dict[str, dict[int, list[float]]] = {}
    for record in records:
        values = (record.get("per_iteration") or {}).get(series)
        if not values:
            continue
        method = str(record.get("method", ""))
        collected.setdefault(method, {})[int(record["seed"])] = [
            float(value) for value in values
        ]
    return collected
