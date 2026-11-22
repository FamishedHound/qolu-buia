"""``hparam_sensitivity.md``: the analysis Chapter 1 describes.

Chapter 1 describes a +/-20 % hyper-parameter sensitivity analysis.  This
document is that analysis: it states, per perturbed
hyper-parameter and direction, what each method scored and whether the
**ranking** the chapter's conclusions rest on survived, and it closes by saying
exactly what the procedure does and does not test.

Rendered from the ``hparam_summary.json`` payload
:mod:`chapter4_qolu.hparam_summary` builds, so ``--report-only`` can rebuild it
from the records without re-running anything.  The summary tables live in
:mod:`chapter4_qolu.reporting.hparam_tables` and the per-run CSV in
:mod:`chapter4_qolu.reporting.hparam_runs`; both are re-exported here, so one
import reaches every document the driver writes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS
from chapter4_qolu.reporting.hparam_runs import (
    HPARAM_RUN_COLUMNS,
    PERTURBATION_COLUMNS,
    render_hparam_runs_csv,
)
from chapter4_qolu.reporting.hparam_tables import (
    ADJUSTMENTS,
    BORROWED_MARK,
    hyperparameter_table,
    number,
    overall_section,
    verdict_table,
)
from chapter4_qolu.reporting.tables import DEFAULT_DECIMALS

__all__ = [
    "HPARAM_RUN_COLUMNS",
    "PERTURBATION_COLUMNS",
    "render_hparam_report",
    "render_hparam_runs_csv",
]


def _intro(summary: Mapping[str, Any]) -> list[str]:
    settings = summary.get("settings") or {}
    configuration = settings.get("configuration") or {}
    percent = float(settings.get("fraction", 0.2)) * 100.0
    return [
        f"# Hyper-parameter sensitivity `{summary.get('tag', '')}`",
        "",
        "## What the thesis promised",
        "",
        "> " + str(summary.get("promise", "")),
        "",
        (
            "No chapter performed one.  This document does, for Chapter 4: "
            f"every key hyper-parameter is moved by +/-{percent:.0f} % **one "
            "at a time**, the comparison is re-run, and the question asked of "
            "each result is the one that sentence actually makes - *does the "
            "method ranking survive?* - rather than whether the numbers move."
        ),
        "",
        f"* **generated**: {summary.get('generated', '')}",
        f"* **raster**: `{settings.get('raster', '')}`",
        f"* **configuration under test**: {configuration.get('description', '')}",
        f"* **window**: `{settings.get('window') or 'automatic densest-land crop'}`",
        f"* **scenarios**: {', '.join(str(k) for k in settings.get('scenarios', []))}",
        f"* **seeds**: {', '.join(str(s) for s in settings.get('seeds', []))}",
        f"* **iterations**: {settings.get('iterations', '')}",
        f"* **methods**: {', '.join(f'`{m}`' for m in settings.get('methods', []))}",
        "",
        str((summary.get("views") or {}).get("note", "")),
        "",
    ]


def _detail_sections(blocks: Sequence[Mapping[str, Any]], decimals: int) -> list[str]:
    lines = ["## Per-perturbation detail", ""]
    if not blocks:
        lines.extend(["_No perturbation has any record on disk yet._", ""])
        return lines
    for block in blocks:
        lines.extend(_detail_section(block, decimals))
    return lines


def _detail_section(block: Mapping[str, Any], decimals: int) -> list[str]:
    lines = [
        f"### `{block.get('label', '')}`",
        "",
        (
            f"{block.get('description', '')} "
            f"(requested {block.get('requested_text', block.get('requested'))}, "
            f"applied {block.get('value_text', block.get('value'))}; effective "
            f"move {float(block.get('applied_fraction') or 0.0) * 100:+.1f} %)."
        ),
        "",
    ]
    flags = [text for name, text in ADJUSTMENTS if block.get(name)]
    if flags:
        lines.extend([f"*Adjustments*: {', '.join(flags)}.", ""])
    if not block.get("scenarios"):
        why = str(block.get("why_not_run", "Not run"))
        lines.extend([why[:1].upper() + why[1:] + ".", ""])
        return lines
    rerun = ", ".join(f"`{m}`" for m in block.get("methods_rerun", []))
    borrowed = ", ".join(f"`{m}`" for m in block.get("methods_borrowed", []))
    lines.extend(
        [
            f"Re-run: {rerun or 'none'}."
            + (f"  Read from the baseline campaign: {borrowed}." if borrowed else ""),
            "",
        ]
    )
    if block.get("inert"):
        why = str(block.get("why_inert", ""))
        lines.extend(
            [
                "**This perturbation changed nothing.**  "
                + why[:1].upper()
                + why[1:]
                + ".  The table below is therefore the baseline's own numbers, "
                "and the verdict that follows it carries no information about "
                "robustness.",
                "",
            ]
        )
    for scenario in block["scenarios"]:
        lines.extend(_scenario_table(scenario, decimals))
    return lines


def _method_row(
    scenario: Mapping[str, Any], method: str, decimals: int
) -> tuple[str, bool]:
    cells = []
    borrowed = False
    for metric in PUBLISHED_METRICS:
        entry = scenario["metrics"][metric]["methods"][method]
        borrowed = bool(entry.get("borrowed_from_baseline"))
        delta = entry.get("delta")
        star = "*" if (entry.get("paired") or {}).get("significant") else ""
        change = (
            ""
            if borrowed or delta is None
            else f" (d {float(delta):+.{decimals}f}{star})"
        )
        cells.append(
            f"{number(entry['mean'], decimals)} +/- "
            f"{number(entry['se'], decimals)}{change}"
        )
    mark = BORROWED_MARK if borrowed else ""
    return f"| `{method}`{mark} | " + " | ".join(cells) + " |", borrowed


def _scenario_table(scenario: Mapping[str, Any], decimals: int) -> list[str]:
    lines = [
        (
            f"**Scenario {scenario.get('scenario')}** - mean +/- SE over "
            "seeds, RAW view; `d` is the change from the baseline "
            "configuration, `*` a paired t-test at p < 0.05 against it."
        ),
        "",
        "| Method | " + " | ".join(f"`{m}`" for m in PUBLISHED_METRICS) + " |",
        "| --- |" + " --- |" * len(PUBLISHED_METRICS),
    ]
    methods = list(scenario["metrics"][PUBLISHED_METRICS[0]]["methods"])
    borrowed_any = False
    for method in methods:
        row, borrowed = _method_row(scenario, method, decimals)
        borrowed_any = borrowed_any or borrowed
        lines.append(row)
    lines.append("")
    if borrowed_any:
        lines.extend(
            [
                (
                    f"{BORROWED_MARK} read from the baseline campaign: this "
                    "perturbation is a field of `QOLUConfig` and cannot reach "
                    "this method, so nothing was re-run and no change is shown."
                ),
                "",
            ]
        )
    for metric in PUBLISHED_METRICS:
        cell = scenario["metrics"][metric]
        verdict = (
            "identical to the baseline" if cell["ranking_preserved"] else "**changed**"
        )
        lines.append(
            f"* `{metric}`: {' > '.join(cell['ranking'])} - {verdict} "
            f"(baseline {' > '.join(cell['baseline_ranking'])}; Kendall tau "
            f"{number(cell['kendall_tau'], 3)})."
        )
    lines.append("")
    return lines


def _method_section(summary: Mapping[str, Any]) -> list[str]:
    settings = summary.get("settings") or {}
    percent = float(settings.get("fraction", 0.2)) * 100.0
    return [
        "## Method - how this analysis was run",
        "",
        (
            "1. The configuration under test is run once as the **baseline**: "
            f"{', '.join(f'`{m}`' for m in settings.get('methods', []))} x "
            f"{len(settings.get('scenarios', []))} scenario(s) x "
            f"{len(settings.get('seeds', []))} seeds."
        ),
        (
            f"2. Each key hyper-parameter is then moved by +/-{percent:.0f} %, "
            "**one at a time**, and the comparison is re-run at the same seeds."
        ),
        (
            "3. An environment-side perturbation changes what every method "
            "sees, so all of them are re-run.  An agent-side one is a field of "
            "`QOLUConfig`, so only the QOLU learner is re-run and the other "
            "methods' baseline records are reused - they would be "
            "bit-identical.  Every borrowed row is flagged, here and in "
            "`hparam_summary.json`."
        ),
        (
            "4. Per cell the methods are ranked by their mean over seeds and "
            "the ranking is compared with the baseline's, exactly and by "
            "Kendall's tau."
        ),
        "",
        "**What this does and does not test.**",
        "",
        (
            "* It perturbs *QOLU's* hyper-parameters and the *shared* "
            "environment quantities, not the baselines' own optimisers, whose "
            "settings are published in their own sources.  For a claim that "
            "QOLU wins, that is the conservative direction: the contribution "
            "is detuned while its comparators stay where their authors put "
            "them."
        ),
        (
            "* It moves one hyper-parameter at a time, so it says nothing "
            "about interactions between them.  The thesis sentence promises "
            'exactly this ("perturbing key hyperparameters"), and a factorial '
            "design over eleven of them is not affordable at 1000-iteration "
            "runs."
        ),
        (
            f"* Where +/-{percent:.0f} % of a value is not expressible - an "
            "integer, an odd window size, a probability that would leave "
            "[0, 1] - the value is rounded half away from zero, nudged away "
            "from the baseline to stay odd, or clamped, and the first table "
            "above names which happened.  A perturbation that lands back on "
            "the baseline is listed, never run and never counted."
        ),
        (
            "* It reports the ranking, not the absolute numbers, because that "
            "is what the promised sentence is about.  The absolute numbers are "
            "all in `hparam_runs.csv`."
        ),
        "",
        (
            "Every number here comes from `hparam_summary.json`; every run "
            "behind them is one row of `hparam_runs.csv` at full precision."
        ),
        "",
    ]


def render_hparam_report(
    summary: Mapping[str, Any], decimals: int = DEFAULT_DECIMALS
) -> str:
    """The whole ``hparam_sensitivity.md`` document."""
    blocks = list(summary.get("perturbations") or [])
    lines = [
        *_intro(summary),
        *hyperparameter_table(list(summary.get("hyperparameters") or [])),
        *overall_section(summary.get("overall") or {}),
        *verdict_table(blocks),
        *_detail_sections(blocks, decimals),
        *_method_section(summary),
    ]
    return "\n".join(lines).rstrip() + "\n"
