"""Writing a campaign's output documents.

``summary.json``      everything the aggregation computed;
``tables.md``         the paper-layout tables, min-max view first;
``tables.tex``        the same tables as an ``\\input``-able LaTeX fragment;
``pattern_check.md``  the ranking comparison against the published tables;
``per_seed.csv``      one full-precision row per run, for re-analysis.

Kept separate from the driver so that a campaign's documents can be rebuilt
from its JSON records alone - no environment, no agent, no raster.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chapter4_qolu.reporting.aggregate import (
    REFERENCE_METHOD,
    aggregate_records,
    load_run_records,
)
from chapter4_qolu.reporting.pattern import render_pattern_check
from chapter4_qolu.reporting.per_seed import render_per_seed_csv
from chapter4_qolu.reporting.tables import (
    DEFAULT_DECIMALS,
    render_latex,
    render_markdown,
)

__all__ = [
    "DEFAULT_DECIMALS",
    "DOCUMENT_NAMES",
    "build_summary",
    "write_campaign_documents",
]

DOCUMENT_NAMES: tuple[str, ...] = (
    "summary.json",
    "tables.md",
    "tables.tex",
    "pattern_check.md",
    "per_seed.csv",
)


def build_summary(
    campaign_dir: Path,
    metadata: Mapping[str, Any] | None = None,
    reference: str = REFERENCE_METHOD,
) -> dict[str, Any]:
    """Load every per-run record of a campaign and aggregate it."""
    records = load_run_records(campaign_dir)
    if not records:
        raise ValueError(f"no per-run records found under {campaign_dir}")
    return aggregate_records(records, reference=reference, metadata=metadata)


def write_campaign_documents(
    campaign_dir: Path,
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]] | None = None,
    decimals: int = DEFAULT_DECIMALS,
) -> dict[str, Path]:
    """Write every campaign document.

    ``records`` supplies the per-run records ``per_seed.csv`` needs; when it is
    omitted they are re-read from ``campaign_dir``.  ``decimals`` controls the
    table precision only - ``summary.json`` and ``per_seed.csv`` always carry
    full double precision.
    """
    directory = Path(campaign_dir)
    directory.mkdir(parents=True, exist_ok=True)
    runs = list(records) if records is not None else load_run_records(directory)
    documents = {
        "summary.json": json.dumps(summary, indent=2),
        "tables.md": render_markdown(summary, decimals=decimals),
        "tables.tex": render_latex(summary, decimals=decimals),
        "pattern_check.md": render_pattern_check(summary),
        "per_seed.csv": render_per_seed_csv(runs),
    }
    written: dict[str, Path] = {}
    for name, text in documents.items():
        path = directory / name
        path.write_text(text, encoding="utf-8")
        written[name] = path
    return written
