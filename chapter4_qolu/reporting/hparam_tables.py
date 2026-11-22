"""The summary tables of ``hparam_sensitivity.md``.

Three of them, in the order a reader needs them:

* **what was perturbed** - every hyper-parameter, its package default, the two
  values actually configured, and any rounding or clamping that took;
* **the headline** - of N effective perturbations, how many preserve the method
  ranking on each published metric, under both views that can differ;
* **perturbation by perturbation** - one line each, with the verdict and
  Kendall's tau.

Section renderers live here and the assembly lives in
:mod:`chapter4_qolu.reporting.hparam_report`, the same split
``sensitivity_report`` makes over ``sensitivity_stage_a`` and
``sensitivity_stage_b``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS
from chapter4_qolu.reporting.tables import DEFAULT_DECIMALS

__all__ = [
    "ADJUSTMENTS",
    "BORROWED_MARK",
    "hyperparameter_table",
    "inclusion_note",
    "number",
    "overall_section",
    "verdict_table",
]

_TICK = "yes"
_CROSS = "**no**"
#: A perturbation that ran and reproduced the baseline exactly; see
#: :mod:`chapter4_qolu.hparam_inert`.
_INERT = "_identical to baseline_"

#: Marks a method whose row was read from the baseline campaign rather than
#: re-run, because the perturbation cannot reach it.
BORROWED_MARK = "&dagger;"

#: How the three adjustment flags are spelled in the tables.
ADJUSTMENTS: tuple[tuple[str, str], ...] = (
    ("rounded", "rounded"),
    ("clamped", "clamped"),
    ("no_op", "identity (not run)"),
)


def number(value: Any, decimals: int = DEFAULT_DECIMALS) -> str:
    """A metric or a tau, rounded for a table; ``-`` when there is none."""
    return "-" if value is None else f"{float(value):.{decimals}f}"


def inclusion_note() -> str:
    """One sentence saying why these eleven and not others."""
    return (
        "A hyper-parameter is included when it is a **scalar** setting of the "
        "learner or of the environment and is either given a number by the "
        "published Table 4.1 or required by the implementation and left open "
        "by the published text.  The architecture rows are excluded - "
        "+/-20 % of a filter count changes the model, not a setting - as are "
        "the switches, which have no +/-20 % and are swept by "
        "`run_sensitivity.py` instead."
    )


def hyperparameter_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """What was perturbed, and what the perturbed values ended up being."""
    if not rows:
        return []
    lines = [
        "## The perturbed hyper-parameters",
        "",
        (
            "`Baseline` is the package default, read off `config.py` rather "
            "than copied here.  The two perturbed columns are the values "
            "actually configured; where the requested value was not "
            "expressible, the adjustment that made it so is named."
        ),
        "",
        "| Hyper-parameter | On | Source | Baseline | Down | Up | Adjustments |",
        "| --- |" + " --- |" * 6,
    ]
    for row in rows:
        adjustments = []
        for direction in ("minus", "plus"):
            item = row.get(direction) or {}
            flags = [text for name, text in ADJUSTMENTS if item.get(name)]
            if flags:
                adjustments.append(f"{direction}: {', '.join(flags)}")
        lines.append(
            "| `{name}` | {target} | {source} | {baseline} | {minus} | {plus} | "
            "{adjustments} |".format(
                name=row.get("name", ""),
                target=row.get("target", ""),
                source=row.get("source", ""),
                baseline=row.get("baseline_text", row.get("baseline", "")),
                minus=(row.get("minus") or {}).get("value_text", "-"),
                plus=(row.get("plus") or {}).get("value_text", "-"),
                adjustments="; ".join(adjustments) or "none",
            )
        )
    lines.extend(["", inclusion_note(), ""])
    return lines


def _inert_note(overall: Mapping[str, Any]) -> list[str]:
    """The paragraph that stops an inert perturbation reading as a survival."""
    n_inert = int(overall.get("n_inert", 0))
    if not n_inert:
        return []
    labels = ", ".join(f"`{label}`" for label in overall.get("inert_labels", []))
    return [
        (
            f"**{n_inert} of the {overall.get('n_effective', 0)} perturbations "
            "that were run changed nothing**: the value was configured and the "
            "runs were performed, but every re-run method came back with "
            "per-seed metrics **identical** to the baseline's"
            + (f" ({labels})" if labels else "")
            + ".  Such a row is excluded from the counts above, because a "
            "ranking that is preserved by construction is not evidence that it "
            "is robust.  The usual cause is a budget the value cannot bind at: "
            "a replay buffer larger than the number of transitions a run "
            "stores never evicts, and a target-sync interval longer than the "
            "number of gradient steps a run takes never fires."
        ),
        "",
    ]


def overall_section(overall: Mapping[str, Any]) -> list[str]:
    """The headline: how many perturbations left the ranking alone."""
    if not overall:
        return []
    total = int(overall.get("n_informative", overall.get("n_effective", 0)))
    headline = (
        (
            "**No perturbation changed the runs at all**, so this analysis "
            "reports no evidence either way about the robustness of the "
            "ranking."
        )
        if total == 0
        else (
            f"**{overall.get('all_metrics_preserved_informative', 0)} / {total}** "
            "of the perturbations that actually changed the runs preserve the "
            "ranking on *all three* published metrics.  Per metric, of "
            f"{total} such perturbations "
            f"({overall.get('n_no_op', 0)} listed perturbation(s) were "
            "arithmetically the identity after rounding or clamping and were "
            f"never run; {overall.get('n_inert', 0)} more were run and "
            "reproduced the baseline exactly):"
        )
    )
    lines = [
        "## Does the method ranking survive?",
        "",
        headline,
        "",
        *_inert_note(overall),
        (
            "| Metric | Ranking preserved (RAW) | mean tau | "
            "Cells preserved | Ranking preserved (absolute) | mean tau |"
        ),
        "| --- |" + " --- |" * 5,
    ]
    for metric in PUBLISHED_METRICS:
        entry = (overall.get("metrics") or {}).get(metric) or {}
        absolute = entry.get("absolute") or {}
        lines.append(
            "| `{metric}` | {preserved} / {total} | {tau} | {cells} / {n_cells} "
            "| {abs_preserved} / {abs_total} | {abs_tau} |".format(
                metric=metric,
                preserved=entry.get("informative_preserved", 0),
                total=entry.get("n_informative", 0),
                tau=number(entry.get("mean_kendall_tau_informative")),
                cells=entry.get("cells_preserved", 0),
                n_cells=entry.get("n_cells", 0),
                abs_preserved=absolute.get("informative_preserved", 0),
                abs_total=absolute.get("n_informative", 0),
                abs_tau=number(absolute.get("mean_kendall_tau_informative")),
            )
        )
    lines.extend(
        [
            "",
            (
                "A perturbation counts as preserving a metric's ranking only "
                "when it does so in **every** scenario it was run in; the "
                '"cells preserved" column counts the (perturbation x '
                "scenario) cells instead, over every effective perturbation "
                "including the inert ones.  Kendall's tau is computed against "
                "the baseline ranking; 1.0 is an identical ordering, -1.0 a "
                "reversed one."
            ),
            "",
        ]
    )
    return lines


def verdict_table(blocks: Sequence[Mapping[str, Any]]) -> list[str]:
    """One line per perturbation (per scenario): preserved, and by how much."""
    if not blocks:
        return []
    lines = [
        "### Perturbation by perturbation",
        "",
        "| Perturbation | Value | Scenario | "
        + " | ".join(f"`{metric}`" for metric in PUBLISHED_METRICS)
        + " |",
        "| --- |" + " --- |" * (2 + len(PUBLISHED_METRICS)),
    ]
    for block in blocks:
        label = block.get("label", "")
        value = block.get("value_text", block.get("value", ""))
        if not block.get("scenarios"):
            lines.append(
                f"| `{label}` | {value} | - |"
                + " not run (identity) |" * len(PUBLISHED_METRICS)
            )
            continue
        if block.get("inert"):
            # Not "yes": the runs were performed and reproduced the baseline,
            # so the ordering is unchanged for a reason that has nothing to do
            # with robustness.  Saying "yes" here would be the whole error.
            lines.append(
                f"| `{label}` | {value} | "
                f"{block['scenarios'][0].get('scenario', '-')} |"
                + f" {_INERT} |" * len(PUBLISHED_METRICS)
            )
            continue
        for scenario in block["scenarios"]:
            cells = []
            for metric in PUBLISHED_METRICS:
                cell = scenario["metrics"][metric]
                mark = _TICK if cell["ranking_preserved"] else _CROSS
                cells.append(f"{mark} (tau {number(cell['kendall_tau'], 2)})")
            lines.append(
                f"| `{label}` | {value} | {scenario['scenario']} | "
                + " | ".join(cells)
                + " |"
            )
    lines.append("")
    return lines
