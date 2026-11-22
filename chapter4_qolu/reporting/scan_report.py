"""``scan.md``: the whole Southeast window scan, winner and all.

The document reports the *distribution*, not only the winner: every retained
window gets a row, the score distribution gets its own table, and the closing
section states the selection rule and the counts it was applied to.

Rendered from the ``scan_summary.json`` payload
:mod:`chapter4_qolu.run_window_scan` builds, so the document can be rebuilt
without re-running anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chapter4_qolu.reporting.scan_stage1 import (
    distribution_table,
    saturation_table,
    stage1_header,
    stage1_table,
)
from chapter4_qolu.reporting.scan_tables import winner_section
from chapter4_qolu.reporting.tables import DEFAULT_DECIMALS

__all__ = ["render_scan_report"]


def _crs_text(crs: Any) -> str:
    """A short name for a CRS, whose ``str`` is often a full WKT document.

    Design note: the first quoted token of a ``PROJCS[...]`` string is taken as
    its name, which is where GDAL puts it; a report that pasted 700 characters
    of WKT into a bullet point would be unreadable.  Anything that does not look
    like WKT is shown as-is.
    """
    if not crs:
        return "no CRS"
    text = str(crs)
    if not text.startswith("PROJCS["):
        return text
    opening = text.find('"')
    closing = text.find('"', opening + 1)
    if opening < 0 or closing < 0:  # pragma: no cover - defensive
        return text
    return text[opening + 1 : closing]


def _box_text(grid: Mapping[str, Any]) -> str:
    """The search box as text, rebuilt from the coordinates when needed."""
    described = grid.get("bounding_box")
    if described:
        return str(described)
    box = grid.get("bounding_box_km") or {}
    if not box:
        return "the search box"
    return (
        f"BNG E {box['east_min']}-{box['east_max']} km, "
        f"N {box['north_min']}-{box['north_max']} km"
    )


def _intro(summary: Mapping[str, Any]) -> list[str]:
    grid = summary["grid"]
    raster = summary.get("raster", {})
    return [
        f"# Southeast window scan `{summary.get('tag', 'se')}`",
        "",
        (
            'The paper places its experiments in "the Southeast UK region, '
            '2015" on a 50 x 50 grid of 1 km^2 parcels but states no extent. '
            "This document scans a grid of candidate 50 x 50 km windows and "
            "reports **every** one of them.  The selection rule is stated "
            'under "Method - how the window was chosen" at the end.'
        ),
        "",
        f"* **generated**: {summary.get('generated', '')}",
        (
            f"* **raster**: `{raster.get('name', '')}` "
            f"({_crs_text(raster.get('crs'))}, "
            f"{raster.get('pixel_size_m', '')} m cells)"
        ),
        f"* **bounding box**: {_box_text(grid)}",
        (
            f"* **window**: {grid['window_size_km']} x "
            f"{grid['window_size_km']} km at a {grid['stride_km']} km stride"
        ),
        (
            "* **sea filter**: windows above "
            f"{float(grid['max_sea_fraction']) * 100:.0f}% sea / no-data are "
            "dropped"
        ),
        (
            f"* **candidate windows**: {grid['n_candidates']} "
            f"({len(grid['eastings'])} eastings x "
            f"{len(grid['northings'])} northings)"
        ),
        f"* **retained**: {grid['n_retained']}",
        "",
        "Easting origins (south-west corner, km): "
        + ", ".join(str(value) for value in grid["eastings"])
        + ".  Northing origins: "
        + ", ".join(str(value) for value in grid["northings"])
        + ".",
        "",
    ]


def _dropped_section(summary: Mapping[str, Any]) -> list[str]:
    dropped = summary.get("dropped") or []
    if not dropped:
        return ["No candidate window exceeded the sea threshold.", ""]
    lines = [
        "## Windows dropped as too marine",
        "",
        "| Window | Extent | Sea / no-data |",
        "| --- | --- | --- |",
    ]
    for item in sorted(dropped, key=lambda entry: entry["window"]["label"]):
        window = item["window"]
        lines.append(
            f"| `{window['label']}` | {window['description']} | "
            f"{float(item['sea_fraction']) * 100:.0f}% |"
        )
    lines.append("")
    return lines


def _stage2_overview(stage2: Mapping[str, Any]) -> list[str]:
    windows = stage2.get("windows") or []
    configuration = stage2.get("configuration", "50x50_f1.0_cap10")
    lines = [
        "## Stage 2 - confirmation",
        "",
        (
            "The top windows re-run over **all four scenarios**, seeds "
            + ", ".join(str(seed) for seed in stage2.get("seeds", []))
            + f", all four methods, configuration `{configuration}`, in the "
            "**weighted** reading (`scale_by_scenario_weight=True`, "
            "`weighted_consensus=True`) - the only reading in which the "
            "scenario weights reach QOLU+BUIA and P-MADDPG at all.  Each "
            "window gets the ordinary campaign documents (`summary.json`, "
            "`tables.md`, `tables.tex`, `pattern_check.md`, `per_seed.csv`)."
        ),
        "",
    ]
    if not windows:
        lines.extend(["No stage-2 records were found.", ""])
        return lines
    lines.extend(
        [
            "| Window | Extent | Cells matching the published order |",
            "| --- | --- | --- |",
        ]
    )
    for entry in windows:
        window = entry["window"]
        lines.append(
            f"| `{window['label']}` | {window['description']} | "
            f"{entry['matched_cells']} of {entry['total_cells']} |"
        )
    lines.append("")
    return lines


def _method_section(
    summary: Mapping[str, Any], windows: Sequence[Mapping[str, Any]] = ()
) -> list[str]:
    selection = summary.get("selection", {})
    winner = selection.get("winner") or {}
    n_candidates = selection.get("n_candidate_windows", 0)
    n_scanned = selection.get("n_scanned_windows", 0)
    name = winner.get("label", "(none)")
    extent = winner.get("description", "")
    best_primary = max(
        (int(entry.get("score", {}).get("first_place_count", 0)) for entry in windows),
        default=0,
    )
    return [
        "## Method - how the window was chosen",
        "",
        (
            f"Windows are compared with the published ordering; every window "
            f"is listed above.  `{name}` ({extent}) is the best of "
            f"{n_scanned} scanned windows out of {n_candidates} candidates "
            f"covering {_box_text(summary.get('grid', {}))}, ranked first by "
            "strict first places, then by Kendall's tau and then by the "
            "min-max margin.  The paper states no extent of its own, so the "
            "window is a property of this scan rather than of the paper."
        ),
        "",
        (
            f"The best primary score reached anywhere in the scan was "
            f"{best_primary} of 3, i.e. the number of Scenario-1 metrics on "
            "which QOLU+BUIA ranks strictly first.  The score distribution "
            f"above gives the full picture across all {n_scanned} windows; the "
            "top window is the maximum of that distribution, so it should be "
            "read together with it."
        ),
        "",
        (
            "The comparison is of the *ordering* only: the published cells are "
            "quoted to two decimals on a 0.05 lattice and every published "
            "standard error is 0.01, 0.02 or 0.03, so the values themselves "
            "are not compared.  The three baselines are re-implementations; "
            "see the README for what each one does."
        ),
        "",
    ]


def render_scan_report(
    summary: Mapping[str, Any], decimals: int = DEFAULT_DECIMALS
) -> str:
    """The full ``scan.md`` document."""
    stage1 = summary.get("stage1", {})
    windows = stage1.get("windows") or []
    lines: list[str] = []
    lines.extend(_intro(summary))
    lines.extend(_dropped_section(summary))
    lines.extend(stage1_header(stage1))
    lines.extend(stage1_table(windows))
    lines.extend(
        distribution_table(
            stage1.get("score_distribution", {}),
            windows,
            stage1.get("configuration", "50x50_f1.0_cap10"),
        )
    )
    lines.extend(saturation_table(windows))
    lines.extend(_stage2_overview(summary.get("stage2", {})))
    lines.extend(winner_section(summary.get("stage2", {}), decimals))
    lines.extend(_method_section(summary, windows))
    return "\n".join(lines)
