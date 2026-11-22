"""The two-panel calibration figure, through the package's figure contract.

Panel (a) is the randomised PIT histogram against the uniform density a
calibrated forecast would produce.  Panel (b) is the PIT reliability diagram -
the empirical CDF of the PIT values against the diagonal - with the empirical
coverage of the central 50 % and 90 % intervals marked on it, so the two
calibration statistics are read off one picture.

Design note (the reliability curve is drawn from the histogram): the PIT values
themselves are not carried into the summary, only their ten-bin histogram, so
the reliability curve has ten points.  That is enough to see a departure from
the diagonal and it keeps the summary JSON a document rather than a data dump;
the underlying appraisals can always be recollected from the checkpoint.

As everywhere in :mod:`chapter4_qolu.figures`, the numbers land on disk as a
CSV whether or not matplotlib is importable, and the PDF and PNG are written
when it is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from chapter4_qolu.figures import backend
from chapter4_qolu.figures.backend import (
    FigureResult,
    new_figure,
    save_figure,
    write_rows,
)

__all__ = ["CALIBRATION_COLUMNS", "calibration_figure"]

CALIBRATION_COLUMNS: tuple[str, ...] = ("panel", "x", "y", "reference", "label")


def _histogram_rows(histogram: dict[str, Any]) -> list[dict[str, Any]]:
    edges = [float(edge) for edge in histogram["edges"]]
    rows: list[dict[str, Any]] = []
    for index, density in enumerate(histogram["density"]):
        lower, upper = edges[index], edges[index + 1]
        rows.append(
            {
                "panel": "pit_histogram",
                "x": repr(0.5 * (lower + upper)),
                "y": repr(float(density)),
                "reference": repr(1.0),
                "label": f"[{lower:.2f},{upper:.2f})",
            }
        )
    return rows


def _reliability_rows(
    histogram: dict[str, Any], coverage: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    edges = [float(edge) for edge in histogram["edges"]]
    counts = np.asarray(histogram["counts"], dtype=np.float64)
    total = counts.sum()
    empirical = np.cumsum(counts) / total if total > 0 else np.zeros_like(counts)
    rows: list[dict[str, Any]] = [
        {
            "panel": "reliability",
            "x": repr(0.0),
            "y": repr(0.0),
            "reference": repr(0.0),
            "label": "pit_cdf",
        }
    ]
    for index, value in enumerate(empirical):
        edge = edges[index + 1]
        rows.append(
            {
                "panel": "reliability",
                "x": repr(edge),
                "y": repr(float(value)),
                "reference": repr(edge),
                "label": "pit_cdf",
            }
        )
    for entry in coverage:
        rows.append(
            {
                "panel": "reliability",
                "x": repr(float(entry["nominal"])),
                "y": repr(float(entry["empirical"])),
                "reference": repr(float(entry["nominal"])),
                "label": f"coverage_{round(100 * float(entry['level']))}",
            }
        )
    return rows


def _draw(histogram: dict[str, Any], coverage: list[dict[str, Any]], title: str) -> Any:
    figure, axes = new_figure(
        1, 2, figsize=(8.4, 3.4), squeeze=False, constrained_layout=True
    )
    edges = np.asarray(histogram["edges"], dtype=float)
    density = np.asarray(histogram["density"], dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])

    left = axes[0][0]
    left.bar(centres, density, width=np.diff(edges), edgecolor="white", linewidth=0.6)
    left.axhline(1.0, linestyle="--", linewidth=1.0, color="0.3")
    left.set_xlabel("randomised PIT $u$", fontsize=8)
    left.set_ylabel("density", fontsize=8)
    left.set_title("(a) PIT histogram (flat = calibrated)", fontsize=9)
    left.tick_params(labelsize=7)

    counts = np.asarray(histogram["counts"], dtype=float)
    total = counts.sum()
    empirical = np.cumsum(counts) / total if total > 0 else np.zeros_like(counts)
    right = axes[0][1]
    right.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="0.3")
    right.plot(
        np.concatenate([[0.0], edges[1:]]),
        np.concatenate([[0.0], empirical]),
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="PIT CDF",
    )
    for entry in coverage:
        right.plot(
            [float(entry["nominal"])],
            [float(entry["empirical"])],
            marker="s",
            markersize=6,
            linestyle="none",
            label=f"{round(100 * float(entry['level']))}% interval coverage",
        )
    right.set_xlabel("nominal", fontsize=8)
    right.set_ylabel("empirical", fontsize=8)
    right.set_title("(b) reliability and interval coverage", fontsize=9)
    right.tick_params(labelsize=7)
    right.legend(fontsize=7, frameon=False, loc="lower right")

    figure.suptitle(title, fontsize=10)
    return figure


def calibration_figure(
    statistics: dict[str, Any],
    out_dir: Path,
    name: str,
    source: str,
    command: str,
    target: str,
) -> FigureResult:
    """Write ``<name>.csv`` (always) and ``<name>.pdf`` / ``.png`` (if drawn)."""
    histogram = statistics["pit"]["histogram"]
    coverage = list(statistics["coverage"])
    rows = _histogram_rows(histogram) + _reliability_rows(histogram, coverage)
    paths = [write_rows(out_dir, name, CALIBRATION_COLUMNS, rows)]
    if backend.matplotlib_available():
        paths.extend(
            save_figure(
                _draw(
                    histogram,
                    coverage,
                    f"QOLU return-distribution calibration ({target})",
                ),
                out_dir,
                name,
            )
        )
    return FigureResult(
        name=name,
        paths=tuple(paths),
        caption=(
            "Calibration of the QOLU categorical return distribution: (a) the "
            "randomised probability integral transform, which is uniform when "
            "the distribution is calibrated, and (b) its reliability diagram "
            "with the empirical coverage of the central 50 % and 90 % "
            f"intervals.  Scored against the {target} reading of the realised "
            "return."
        ),
        source=source,
        command=command,
        notes=(
            "the PIT is randomised because the C51 support is discrete",
            "the reliability curve is the cumulative PIT histogram, ten points",
        ),
    )
