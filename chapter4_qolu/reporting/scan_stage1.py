"""The stage-1 screening section of ``scan.md``.

One row per retained window - composition, the three scores, how many metrics
separated no methods at all, and the window's ordering on each published
metric with tied methods joined by ``=`` - followed by the distribution of the
primary score and by the saturation / conversion table, which at the published
scale is where the methods actually differ.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS
from chapter4_qolu.reporting.paths import NO_CAP
from chapter4_qolu.reporting.tables import METHOD_LABELS, ordered_methods
from chapter4_qolu.window_scan import class_share_text

#: Trailing ``cap<token>`` component of a configuration label, e.g. ``cap10``
#: or ``capnone`` (see ``chapter4_qolu.reporting.paths.ScaleConfig.label``).
_CAP_TOKEN = re.compile(r"cap(\w+)\Z")

__all__ = [
    "distribution_table",
    "saturation_table",
    "stage1_header",
    "stage1_table",
]


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _order_text(ranking: Sequence[str]) -> str:
    return " > ".join(_label(method) for method in ranking)


def _group_text(groups: Sequence[Sequence[str]]) -> str:
    """The order with tied methods joined by ``=`` instead of ``>``."""
    return " > ".join(
        " = ".join(_label(method) for method in group) for group in groups
    )


def _cap_phrase(configuration: str) -> str:
    """Prose for a configuration label's per-iteration conversion cap.

    ``50x50_f1.0_cap10`` -> ``"up to 10 conversions per iteration"``;
    ``50x50_f1.0_capnone`` -> ``"no cap"``.
    """
    match = _CAP_TOKEN.search(str(configuration))
    token = match.group(1) if match else None
    if not token or token.lower() == NO_CAP:
        return "no cap"
    return f"up to {token} conversions per iteration"


def stage1_header(stage1: Mapping[str, Any]) -> list[str]:
    return [
        "## Stage 1 - screening",
        "",
        (
            f"All four methods on **Scenario 1**, seed 0, "
            f"{stage1.get('iterations', 1000)} iterations, configuration "
            f"`{stage1.get('configuration', '50x50_f1.0_cap10')}`, "
            "`--qolu-reward-mode "
            f"{stage1.get('qolu_reward_mode', 'per_parcel')}`, "
            "the literal (unweighted) reading "
            "of the scenario weights.  Scenario 1 weights the three "
            "stakeholders equally, so the literal and weighted readings "
            "coincide there and the screening does not depend on which one is "
            "preferred."
        ),
        "",
        (
            "Scores, in lexicographic priority: **first** the number of the "
            "three published metrics on which QOLU+BUIA ranks *strictly* first "
            "(0-3, `First`); **then** the sum of Kendall's tau-a between the "
            "rebuild's **values** and the published "
            "QOLU+BUIA > P-MADDPG > LToS > MitM (+3 = the published order on "
            "every metric, -3 = its exact reverse, `Tau`); **then** "
            "QOLU+BUIA's mean cross-method min-max score (`Min-max`)."
        ),
        "",
        (
            "**Ties are the normal case here, not an edge case**, so every "
            "number above is computed from the metric values and never from a "
            "ranking.  A tied pair of methods counts neither for nor against "
            "the published order; a shared top place is not a first place "
            "(the published claim is a strict win) and is reported separately "
            "in `Joint`; a metric on which *every* method scored the same is "
            "counted in `Flat`, scores tau = 0 and puts nobody first.  The "
            "alternative - ranking the methods and breaking ties by name, "
            "which is what the campaign tables do for reproducibility - would "
            "hand every tie to `ltos` and count it against `qolu_buia`, purely "
            "because of how the two sort."
        ),
        "",
    ]


def stage1_table(windows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        (
            "| # | Window | Extent | Sea | Dominant cover | First | Joint | "
            "Tau | Min-max | Flat | Land Use order | Woodland order | "
            "Urban order |"
        ),
        "| --- |" + " --- |" * 12,
    ]
    for index, entry in enumerate(windows, start=1):
        window = entry["window"]
        composition = entry["composition"]
        score = entry["score"]
        per_metric = {item["metric"]: item for item in score.get("metrics", [])}
        cells = []
        for metric in PUBLISHED_METRICS:
            item = per_metric.get(metric)
            if item is None:
                cells.append("-")
            elif item.get("flat"):
                cells.append("(flat: all four identical)")
            elif item.get("groups"):
                cells.append(_group_text(item["groups"]))
            else:
                cells.append(_order_text(item["ranking"]))
        lines.append(
            "| {index} | `{label}` | {extent} | {sea:.0f}% | {cover} | {first}/3 "
            "| {joint}/3 | {tau:+.3f} | {minmax:.3f} | {flat} | {a} | {b} | "
            "{c} |".format(
                index=index,
                label=window["label"],
                extent=window["description"],
                sea=float(composition["sea_fraction"]) * 100.0,
                cover=class_share_text(composition["class_shares"]),
                first=int(score["first_place_count"]),
                joint=int(score.get("joint_first_count", 0)),
                tau=float(score["kendall_sum"]),
                minmax=float(score["reference_minmax_mean"]),
                flat=int(score.get("flat_metrics", 0)),
                a=cells[0],
                b=cells[1],
                c=cells[2],
            )
        )
    lines.append("")
    return lines


def distribution_table(
    distribution: Mapping[str, Any],
    windows: Sequence[Mapping[str, Any]],
    configuration: str = "50x50_f1.0_cap10",
) -> list[str]:
    total = len(windows)
    lines = [
        "### Score distribution",
        "",
        "| QOLU+BUIA ranks first on ... of the 3 metrics | Windows | Share |",
        "| --- | --- | --- |",
    ]
    for count in sorted(distribution, key=int):
        value = int(distribution[count])
        share = (value / total * 100.0) if total else 0.0
        lines.append(f"| {count} | {value} | {share:.0f}% |")
    lines.append("")

    degenerate = sum(
        1
        for entry in windows
        if int(entry.get("score", {}).get("flat_metrics", 0)) == len(PUBLISHED_METRICS)
    )
    if total and degenerate:
        lines.extend(
            [
                (
                    f"On **{degenerate} of the {total} scanned windows all four "
                    "methods produced identical values on all three metrics**.  "
                    "That is not a bug in the comparison: at the published "
                    f"scale a 50 x 50 grid with {_cap_phrase(configuration)} is "
                    "fully developed long before iteration 1000, "
                    "and the three published metrics are functions of the final "
                    "landscape, which is then the same whichever policy got "
                    "there.  Where the methods do differ it is because one of "
                    "them stopped short of developing everything.  The "
                    "`Saturation iteration` table below is where the difference "
                    "between the methods actually shows up at this scale."
                ),
                "",
            ]
        )
    elif total:
        lines.extend(
            [
                (
                    f"On **{degenerate} of the {total} scanned windows all four "
                    "methods produced identical values on all three metrics**: "
                    f"at configuration `{configuration}` the methods differ on "
                    "every scanned window.  The `Saturation iteration` table "
                    "below shows how many parcels each method converted."
                ),
                "",
            ]
        )
    return lines


def _below_half_counts(
    windows: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> dict[str, int]:
    """Per method, the number of windows where its total conversions were
    below 50% of the best-converting method's total conversions *in that
    window* - the same records the saturation table itself renders.
    """
    counts = {method: 0 for method in methods}
    for entry in windows:
        blocks = entry.get("methods", {}) or {}
        totals = {
            method: blocks[method]["total_conversions"]
            for method in methods
            if blocks.get(method)
            and blocks[method].get("total_conversions") is not None
        }
        if not totals:
            continue
        best = max(totals.values())
        if best <= 0:
            continue
        for method, value in totals.items():
            if value < 0.5 * best:
                counts[method] += 1
    return counts


def _below_half_sentence(
    windows: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> str:
    counts = _below_half_counts(windows, methods)
    parts = [f"{counts[method]} ({_label(method)})" for method in methods]
    counts_text = (
        parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"
    )
    return (
        "Total conversions were below 50% of the best-converting method's "
        f"on {counts_text} of the {len(windows)} windows."
    )


def saturation_table(windows: Sequence[Mapping[str, Any]]) -> list[str]:
    methods = ordered_methods(
        {method for entry in windows for method in entry.get("methods", {})}
    )
    if not methods:
        return []
    header = " | ".join(_label(method) for method in methods)
    lines = [
        "### Saturation iteration and total conversions (stage 1)",
        "",
        f"| Window | {header} |",
        "| --- |" + " --- |" * len(methods),
    ]
    for entry in windows:
        cells = []
        for method in methods:
            block = entry.get("methods", {}).get(method)
            if not block:
                cells.append("-")
                continue
            saturation = block.get("saturation_iteration")
            cells.append(
                f"{'never' if saturation is None else saturation} / "
                f"{block.get('total_conversions', '-')}"
            )
        lines.append(f"| `{entry['window']['label']}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Cells are *saturation iteration / total conversions*.",
            "",
            _below_half_sentence(windows, methods),
            "",
        ]
    )
    return lines
