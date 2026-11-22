"""The stage-A screening section of ``sensitivity.md``.

One row per configuration - **all** of them, best first - with its best window,
the three scan scores, the value-closeness statistic on both candidate views,
and the configuration's ordering on each published metric with tied methods joined by
``=``.  Then the distribution of the primary score, which is the part that
actually answers the question the sweep was asked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS
from chapter4_qolu.reporting.tables import METHOD_LABELS

__all__ = [
    "distribution_table",
    "restraint_note",
    "stage_a_header",
    "stage_a_table",
    "top_table",
]

#: The method the published tables call "US".
_REFERENCE = "qolu_buia"


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _group_text(groups: Sequence[Sequence[str]]) -> str:
    return " > ".join(
        " = ".join(_label(method) for method in group) for group in groups
    )


def _order_cells(score: Mapping[str, Any]) -> list[str]:
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
            cells.append(" > ".join(_label(m) for m in item.get("ranking", [])))
    return cells


def _number(value: Any, decimals: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{decimals}f}"


def stage_a_header(stage_a: Mapping[str, Any]) -> list[str]:
    axes = stage_a.get("axes", {})
    windows = ", ".join(
        f"`{item['label']}` ({item['description']})"
        for item in stage_a.get("windows", [])
    )
    return [
        "## Stage A - screening",
        "",
        (
            "All four methods on **Scenario 1**, seed "
            f"{stage_a.get('seed', 0)}, {stage_a.get('iterations', 1000)} "
            "iterations, the literal (unweighted) reading of the scenario "
            "weights.  Scenario 1 weights the three stakeholders equally, so "
            "the literal and weighted readings coincide there and the "
            "screening does not depend on which one is preferred."
        ),
        "",
        "The sweep is over the reconstruction's free choices:",
        "",
        (
            "* **environment**: "
            + ", ".join(f"`{v}`" for v in axes.get("environment", []))
        ),
        f"* **conversion cap**: {', '.join(str(v) for v in axes.get('cap', []))}",
        (
            "* **QOLU reward reading**: "
            + ", ".join(f"`{v}`" for v in axes.get("qolu_reward_mode", []))
        ),
        (
            "* **P-MADDPG reconstruction**: "
            + ", ".join(f"`{v}`" for v in axes.get("pmaddpg", []))
        ),
        "* **LToS variant**: " + ", ".join(f"`{v}`" for v in axes.get("ltos", [])),
        "* **MitM variant**: " + ", ".join(f"`{v}`" for v in axes.get("mitm", [])),
        "",
        f"Environments: {windows}.",
        "",
        (
            f"That is **{stage_a.get('n_configurations', 0)} configurations**, "
            f"{stage_a.get('n_cells', 0)} scored cells.  Every one of them is "
            "in the table below; none is hidden."
        ),
        "",
        (
            "Scores, in lexicographic priority: **First**, the number of the "
            "three published metrics on which QOLU+BUIA ranks *strictly* first "
            "(0-3); **Tau**, the sum of Kendall's tau-a between the rebuild's "
            "values and the published QOLU+BUIA > P-MADDPG > LToS > MitM (+3 = "
            "the published order on every metric, -3 = its exact reverse); "
            "**MAE**, the mean absolute difference from the published Table-4 "
            "cells on view ("
            f"{stage_a.get('closeness_view', 'absolute')}"
            "), smaller being closer; and **Min-max**, QOLU+BUIA's mean "
            "cross-method min-max score, as the final margin."
        ),
        "",
        (
            "`MAE(c)` is the same statistic on the cross-method min-max view, "
            "printed beside it so the two cannot be confused.  Both are "
            "ranking statistics that order the configurations against each "
            'other; see "Published normalisation" above for which view each '
            "one is computed on."
        ),
        "",
        (
            "Every number is computed from the metric **values**, never from a "
            "ranking: a tied pair counts neither for nor against the published "
            "order, a shared top place is not a first place, and a metric on "
            "which all four methods agree is counted in `Flat`."
        ),
        "",
        (
            "**At cap 1, total conversions do not separate the methods.**  The "
            "revised P-MADDPG explores with epsilon-greedy 0.3 per candidate, "
            "so at a one-parcel budget some candidate is always approved and "
            "every method converts exactly 1000 parcels.  What separates them "
            "there is *which* parcels convert, which is what the three metrics "
            "measure and what the before/after maps show; the conversion "
            "column is informative only at the larger caps."
        ),
        "",
    ]


def _row(index: int, entry: Mapping[str, Any]) -> str:
    configuration = entry["configuration"]
    score = entry["score"]
    cells = _order_cells(score)
    return (
        "| {index} | `{label}` | `{window}` | {first}/3 | {joint}/3 | {tau:+.3f} "
        "| {mae} | {mae_c} | {minmax:.3f} | {flat} | {n} | {a} | {b} | {c} |".format(
            index=index,
            label=configuration["label"],
            window=configuration.get("environment", entry["window"]["label"]),
            first=int(score["first_place_count"]),
            joint=int(score.get("joint_first_count", 0)),
            tau=float(score["kendall_sum"]),
            mae=_number(entry.get("value_mae")),
            mae_c=_number(entry.get("value_mae_minmax")),
            minmax=float(score["reference_minmax_mean"]),
            flat=int(score.get("flat_metrics", 0)),
            n=int(entry.get("n_records", 0)),
            a=cells[0],
            b=cells[1],
            c=cells[2],
        )
    )


def stage_a_table(configurations: Sequence[Mapping[str, Any]]) -> list[str]:
    """One row per configuration, best first - the whole distribution."""
    lines = [
        (
            "| # | Configuration | Environment | First | Joint | Tau | MAE | "
            "MAE(c) | Min-max | Flat | n | Land Use order | Woodland order | "
            "Urban order |"
        ),
        "| --- |" + " --- |" * 13,
    ]
    for index, entry in enumerate(configurations, start=1):
        lines.append(_row(index, entry))
    lines.append("")
    return lines


def top_table(
    configurations: Sequence[Mapping[str, Any]], how_many: int = 5
) -> list[str]:
    """The best few configurations spelled out axis by axis."""
    if not configurations:
        return []
    lines = [
        "### The best configurations, by coordinate",
        "",
        (
            "| # | Environment | Cap | QOLU reward | P-MADDPG | LToS | MitM | "
            "First | Tau | MAE |"
        ),
        "| --- |" + " --- |" * 9,
    ]
    for index, entry in enumerate(configurations[:how_many], start=1):
        configuration = entry["configuration"]
        score = entry["score"]
        lines.append(
            "| {index} | `{env}` | {cap} | `{mode}` | `{pmaddpg}` | `{ltos}` "
            "| `{mitm}` | {first}/3 | {tau:+.3f} | {mae} |".format(
                index=index,
                env=configuration.get("environment", "?"),
                cap=configuration.get("cap"),
                mode=configuration.get("reward_mode"),
                pmaddpg=configuration.get("pmaddpg"),
                ltos=configuration.get("ltos"),
                mitm=configuration.get("mitm"),
                first=int(score["first_place_count"]),
                tau=float(score["kendall_sum"]),
                mae=_number(entry.get("value_mae")),
            )
        )
    lines.append("")
    return lines


def restraint_note(configurations: Sequence[Mapping[str, Any]]) -> list[str]:
    """Say so when the best configuration wins by *converting less*.

    The three published metrics are all functions of the final landscape, and
    two of them reward leaving it alone: a method that approves fewer parcels
    preserves more woodland and keeps what it does build nearer the existing
    town.  So "QOLU+BUIA ranks first" can mean either that it *placed*
    development better or simply that it *did less of it*, and the two are
    worth telling apart before anything is read into a first place.  This is
    computed from the cell's own conversion counts, not asserted.
    """
    if not configurations:
        return []
    best = configurations[0]
    methods = best.get("methods") or {}
    counts = {
        name: int(block["total_conversions"])
        for name, block in methods.items()
        if block.get("total_conversions") is not None
    }
    reference = counts.get(_REFERENCE)
    others = {name: value for name, value in counts.items() if name != _REFERENCE}
    if reference is None or not others:
        return []
    fewest_other = min(others.values())
    if reference >= fewest_other:
        return []
    shortfall = 100.0 * (1.0 - reference / fewest_other)
    return [
        "### How the leading configuration wins",
        "",
        (
            f"In `{best['configuration']['label']}`, QOLU+BUIA converted "
            f"**{reference}** parcels against {fewest_other}-{max(others.values())} "
            f"for the three baselines - **{shortfall:.0f}% fewer than the most "
            "restrained of them**.  Two of the three published metrics reward "
            "leaving the landscape alone (Woodland is the share of the initial "
            "woodland and arable area still standing; Urban is closeness to "
            "*existing* development), so a first place here is at least partly "
            "a *restraint* effect rather than a placement effect.  The "
            "before/after maps and the `region_converted_*` figure show which "
            "parcels each method took, which is the part a metric of the final "
            "landscape cannot express."
        ),
        "",
    ]


def distribution_table(
    distribution: Mapping[str, Any], configurations: Sequence[Mapping[str, Any]]
) -> list[str]:
    total = len(configurations)
    lines = [
        "### Distribution of the primary score",
        "",
        "| QOLU+BUIA ranks first on ... of the 3 metrics | Configurations | Share |",
        "| --- | --- | --- |",
    ]
    for count in sorted(distribution, key=int):
        value = int(distribution[count])
        share = (value / total * 100.0) if total else 0.0
        lines.append(f"| {count} | {value} | {share:.0f}% |")
    lines.append("")
    best = max(
        (int(entry["score"]["first_place_count"]) for entry in configurations),
        default=0,
    )
    lines.extend(
        [
            (
                f"**The best primary score reached anywhere in the sweep was "
                f"{best} of 3.**  The environment is one of the swept "
                "coordinates, so each row of the table above is one "
                "configuration and nothing is aggregated away."
            ),
            "",
        ]
    )
    return lines
