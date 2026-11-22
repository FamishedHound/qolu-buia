"""The markdown document of a diagnostics sweep: the *aggregated* numbers.

Same shape as :mod:`chapter4_qolu.diagnostics.report`, one level up: what was
run, the per-cell table, the aggregate across cells, the split by scenario and
by stakeholder objective, and a verdict written so that it reads the same way
whichever way it comes out.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

__all__ = ["DEFAULT_DECIMALS", "render_sweep_report"]

DEFAULT_DECIMALS: int = 4

#: Column label for every averaged field, in reporting order.
_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("n", "appraisals scored"),
    ("effective_n", "independent observations (effective n)"),
    ("skill_climatology", "CRPS skill vs climatology"),
    ("skill_initialisation", "CRPS skill vs initialisation"),
    ("crps", "CRPS"),
    ("crps_climatology", "CRPS of climatology"),
    ("ks_statistic", "PIT KS statistic"),
    ("ks_p_value", "PIT KS p-value"),
    ("coverage_50", "coverage of the central 50 % interval"),
    ("width_50", "mean width of the central 50 % interval"),
    ("coverage_90", "coverage of the central 90 % interval"),
    ("width_90", "mean width of the central 90 % interval"),
    ("mae", "MAE of E[Z]"),
    ("rmse", "RMSE of E[Z]"),
    ("bias", "bias of E[Z] (predicted - realised)"),
    ("sd_realised", "SD of the realised return"),
    ("boundary_mass", "boundary-atom mass"),
    ("out_of_support_fraction", "realised returns outside the support"),
    ("realised_sd_in_atoms", "SD of the realised return, in atoms"),
    ("distinct_atoms_used", "atoms the realised returns land on"),
)

#: The per-cell table, which has to fit on a page.
_CELL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("skill_climatology", "CRPS skill vs clim."),
    ("skill_initialisation", "vs init."),
    ("ks_statistic", "PIT KS"),
    ("coverage_50", "cov 50 %"),
    ("coverage_90", "cov 90 %"),
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("boundary_mass", "boundary mass"),
)


def _number(value: Any, decimals: int) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.{decimals}f}"


def _mean_se(entry: dict[str, Any] | None, decimals: int) -> str:
    if not entry:
        return "-"
    return f"{_number(entry['mean'], decimals)} ± {_number(entry['se'], decimals)}"


def _cell_table(cells: list[dict[str, Any]], target: str, decimals: int) -> list[str]:
    header = "| Scenario | Seed | " + " | ".join(label for _, label in _CELL_COLUMNS)
    lines = [header + " |", "| --- | --- |" + " --- |" * len(_CELL_COLUMNS)]
    for cell in sorted(cells, key=lambda item: (item["scenario"], item["seed"])):
        flat = cell["targets"].get(target, {})
        values = " | ".join(
            _number(flat.get(key), decimals) for key, _ in _CELL_COLUMNS
        )
        lines.append(f"| {cell['scenario']} | {cell['seed']} | {values} |")
    return lines


def _aggregate_table(aggregate: dict[str, Any], decimals: int) -> list[str]:
    lines = ["| Statistic | mean ± SE | SD | n |", "| --- | --- | --- | --- |"]
    for key, label in _FIELD_LABELS:
        entry = aggregate.get(key)
        if entry is None:
            continue
        lines.append(
            f"| {label} | {_mean_se(entry, decimals)} | "
            f"{_number(entry['sd'], decimals)} | {entry['n']} |"
        )
    return lines


def _grouped_table(
    groups: dict[str, Any], first: str, decimals: int, keys: tuple[str, ...]
) -> list[str]:
    labels = dict(_FIELD_LABELS)
    header = f"| {first} | " + " | ".join(labels[key] for key in keys) + " |"
    lines = [header, "| --- |" + " --- |" * len(keys)]
    for name, aggregate in groups.items():
        cells = " | ".join(_mean_se(aggregate.get(key), decimals) for key in keys)
        lines.append(f"| {name} | {cells} |")
    return lines


def _settings_line(summary: dict[str, Any]) -> str:
    run = summary.get("settings", {})
    return (
        f"Method `{run.get('method', '-')}`, window `{run.get('window', '-')}`, "
        f"configuration `{run.get('configuration', '-')}`, reward reading "
        f"`{run.get('qolu_reward_mode', '-')}`, "
        f"{run.get('stakeholders', '-')} stakeholders, "
        f"{run.get('training_iterations', '-')} training iterations, "
        f"{run.get('evaluation_iterations', '-')} evaluation iterations, "
        f"scenarios {run.get('scenarios', '-')}, seeds {run.get('seeds', '-')}."
    )


def render_sweep_report(
    summary: dict[str, Any], decimals: int = DEFAULT_DECIMALS
) -> str:
    """Render the sweep summary as one markdown document."""
    target = summary.get("headline_target", "tail_return")
    generated = summary.get(
        "generated", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    )
    cells = list(summary.get("cells", []))
    verdict = summary.get("verdict", {})

    lines: list[str] = [
        "# QOLU return-distribution diagnostics across scenarios and seeds",
        "",
        f"Generated {generated}.  Tag `{summary.get('tag', '-')}`.",
        "",
        _settings_line(summary),
        "",
        (
            f"{summary.get('n_complete', 0)} of {summary.get('n_cells', 0)} cells "
            f"finished.  Every number below is scored against `{target}`; the "
            "other two realised-return readings are in the per-cell JSON beside "
            "this file, and are never mixed with this one."
        ),
        "",
    ]
    if summary.get("missing"):
        lines += [
            "**Cells missing from this aggregation:** "
            + ", ".join(f"`{tag}`" for tag in summary["missing"])
            + ".",
            "",
        ]
    if not cells:
        lines += [verdict.get("statement", "Nothing has finished."), ""]
        return "\n".join(lines) + "\n"

    lines += ["## 1. Per cell", "", *_cell_table(cells, target, decimals), ""]
    lines += [
        "## 2. Aggregated over every cell",
        "",
        *_aggregate_table(summary.get("overall", {}), decimals),
        "",
        (
            "The spread is across scenarios *and* seeds, so it is the spread a "
            "chapter claim has to survive, not a within-run standard error."
        ),
        "",
        "## 2b. The same statistics against each realised-return reading",
        "",
        *_grouped_table(
            summary.get("by_target", {}),
            "Realised return",
            decimals,
            (
                "skill_climatology",
                "skill_initialisation",
                "ks_statistic",
                "coverage_50",
                "coverage_90",
                "mae",
                "rmse",
                "sd_realised",
                "realised_sd_in_atoms",
            ),
        ),
        "",
        (
            "`n_step_target` is the only one of the three the network is "
            "literally regressed onto, so read its row as the fit to the "
            "training target and `tail_return` as the fit to the long-run "
            "return.  The three live on different scales and are never mixed."
        ),
        "",
        "## 3. By scenario",
        "",
        *_grouped_table(
            summary.get("per_scenario", {}),
            "Scenario",
            decimals,
            ("skill_climatology", "ks_statistic", "coverage_90", "mae", "rmse"),
        ),
        "",
        "## 4. By stakeholder objective",
        "",
        *_grouped_table(
            summary.get("per_objective", {}),
            "Objective",
            decimals,
            (
                "skill_climatology",
                "skill_initialisation",
                "ks_statistic",
                "coverage_50",
                "coverage_90",
                "mae",
                "rmse",
                "boundary_mass",
            ),
        ),
        "",
        "## 5. Verdict",
        "",
        verdict.get("statement", "-"),
        "",
        verdict.get("calibration_caveat", ""),
        "",
        "| Question | Cells |",
        "| --- | --- |",
        (
            "| Is `Z` state-conditional (positive **CRPS skill** vs "
            f"climatology)? | {verdict.get('n_state_conditional', 0)} of "
            f"{verdict.get('n_cells', 0)} |"
        ),
        (
            "| Is `Z` calibrated (randomised PIT consistent with uniform)? | "
            f"{verdict.get('n_calibrated', 0)} of {verdict.get('n_cells', 0)} |"
        ),
        (
            "| Is the atom support adequate (low **boundary**-atom mass)? | "
            f"{verdict.get('n_support_adequate', 0)} of "
            f"{verdict.get('n_cells', 0)} |"
        ),
        (
            "| Is the atom spacing fine enough to express the signal? | "
            f"{verdict.get('n_resolves_the_signal', 0)} of "
            f"{verdict.get('n_cells', 0)} |"
        ),
        "",
        (
            "The policy row is not repeated here: the decision path reads `Z` "
            "only through `E[Z]` by construction, which is a property of the "
            "code and not of any run.  See any cell's own "
            "`distribution_diagnostics.md`, section 1."
        ),
        "",
    ]
    return "\n".join(lines) + "\n"
