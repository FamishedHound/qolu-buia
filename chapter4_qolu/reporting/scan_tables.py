"""The winning window's own tables inside ``scan.md``.

Stage 2 already writes the ordinary campaign documents per window; this module
reproduces the closest window's numbers *inside* the scan report, at finer
precision than the published tables and with the spread and the test the
published tables leave out:

* view (c) cross-method min-max, ``mean +- SE`` - the analogue of the paper's
  "normalized between 0 and 1";
* view (a) RAW, ``mean +- SD`` with the seed count ``n``, because an SE alone
  hides how many runs it came from;
* the paired t-test of QOLU + BUIA against P-MADDPG, with ``t``, ``p`` and the
  degrees of freedom.  P-MADDPG is the comparison because the published tables
  put it second in every scenario and tie it with QOLU + BUIA away from the
  emphasised metric, so it is the ordering claim that actually has to be tested.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS, REFERENCE_METHOD
from chapter4_qolu.reporting.tables import (
    DEFAULT_DECIMALS,
    METHOD_LABELS,
    METRIC_LABELS,
    ordered_methods,
)

__all__ = ["COMPARISON_METHOD", "scenario_tables", "test_lines", "winner_section"]

#: The baseline the winning window's paired tests are reported against.
COMPARISON_METHOD = "pmaddpg"


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _number(value: Any, decimals: int = DEFAULT_DECIMALS, dash: str = "-") -> str:
    if value is None:
        return dash
    return f"{float(value):.{decimals}f}"


def scenario_tables(
    scenario: Mapping[str, Any], decimals: int = DEFAULT_DECIMALS
) -> list[str]:
    methods = ordered_methods(scenario["methods"])
    lines = [
        f"#### Scenario {scenario['scenario']}",
        "",
        "| Method | " + " | ".join(METRIC_LABELS[m] for m in PUBLISHED_METRICS) + " |",
        "| --- | --- | --- | --- |",
    ]
    for method in methods:
        block = scenario["methods"][method].get("minmax", {})
        cells = []
        for metric in PUBLISHED_METRICS:
            entry = block.get(metric)
            cells.append(
                "-"
                if entry is None
                else f"{_number(entry['mean'], decimals)} ± "
                f"{_number(entry.get('se'), decimals)}"
            )
        lines.append(f"| {_label(method)} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            (
                "*view (c) cross-method min-max, mean ± SE over seeds.  The "
                "SE is the RAW one divided by the same span, so a value near "
                "1 means the spread across seeds is as wide as the whole gap "
                "between the best and the worst method - read the RAW table "
                "below before reading anything into the ordering.*"
            ),
            "",
            "| Method | "
            + " | ".join(METRIC_LABELS[m] for m in PUBLISHED_METRICS)
            + " | n |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for method in methods:
        block = scenario["methods"][method].get("raw", {})
        cells = []
        seeds = 0
        for metric in PUBLISHED_METRICS:
            entry = block.get(metric)
            if entry is None:
                cells.append("-")
                continue
            seeds = max(seeds, int(entry.get("n", 0)))
            cells.append(
                f"{_number(entry['mean'], decimals)} ± "
                f"{_number(entry.get('sd'), decimals)}"
            )
        lines.append(f"| {_label(method)} | " + " | ".join(cells) + f" | {seeds} |")
    lines.extend(["", "*view (a) RAW, mean ± SD over seeds.*", ""])
    lines.extend(test_lines(scenario, decimals))
    return lines


def test_lines(
    scenario: Mapping[str, Any], decimals: int = DEFAULT_DECIMALS
) -> list[str]:
    tests = (scenario.get("tests") or {}).get(COMPARISON_METHOD)
    if not tests:
        return []
    parts = []
    for metric in PUBLISHED_METRICS:
        entry = tests.get(metric)
        if not entry:
            continue
        if not entry.get("possible"):
            parts.append(f"{METRIC_LABELS[metric]}: {entry.get('note', 'no test')}")
            continue
        parts.append(
            f"{METRIC_LABELS[metric]}: t = {_number(entry.get('t'), decimals)}, "
            f"p = {_number(entry.get('p_value'), decimals)}, "
            f"df = {entry.get('df')}"
        )
    if not parts:
        return []
    return [
        "Paired t-test, "
        f"{_label(REFERENCE_METHOD)} vs {_label(COMPARISON_METHOD)} on the RAW "
        "view (positive t = QOLU+BUIA higher): " + "; ".join(parts) + ".",
        "",
    ]


def winner_section(
    stage2: Mapping[str, Any], decimals: int = DEFAULT_DECIMALS
) -> list[str]:
    windows = stage2.get("windows") or []
    if not windows:
        return []
    entry = windows[0]
    window = entry["window"]
    lines = [
        f"### Closest window: `{window['label']}` ({window['description']})",
        "",
        (
            f"{entry['matched_cells']} of {entry['total_cells']} "
            "(scenario x metric) cells reproduce the published ordering."
        ),
        "",
    ]
    for configuration in entry.get("summary", {}).get("configurations", []):
        for scenario in configuration.get("scenarios", []):
            lines.extend(scenario_tables(scenario, decimals))
    return lines
