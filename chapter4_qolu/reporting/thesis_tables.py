"""Thesis-ready LaTeX for Chapter 4, written from files a run already left.

Two families, each with a plain-text note beside it that states what every
number is - the view, the sample, the test - so the prose can quote a table
without re-deriving anything:

* one results table per scenario, from a campaign's ``summary.json``: a row per
  method in the table order (the ``no_regulator`` ablation first), columns
  Land Use, Woodland, Urban, ``C_align`` and Conversions, ``mean ± SE`` over
  seeds, the best value of a column in bold, and a marker on a baseline whose
  paired t-test against QOLU+BUIA - computed by the aggregation on the view
  the table shows - has ``p < 0.05``;
* the distribution-accuracy table, from a diagnostics sweep's
  ``distribution_diagnostics_sweep.json``: per scenario and pooled.

Nothing is re-tested here.  The paired tests are read out of ``summary.json``
(``tests_by_view``), and the distribution rows out of the sweep's per-cell
records; the only arithmetic is the mean and standard error across cells, and
the average over a cell's stakeholders for the two policy quantities, both
stated in the note.

Design note (booktabs, no colour): ``\\toprule`` / ``\\midrule`` /
``\\bottomrule`` and no vertical rules, so the fragments need ``booktabs`` and,
for ``\\text`` in the ``C_align`` header, ``amsmath``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chapter4_qolu.reporting.stats import summarise
from chapter4_qolu.reporting.tables import METHOD_LABELS, ordered_methods

__all__ = [
    "DISTRIBUTION_ROWS",
    "SCENARIO_COLUMNS",
    "SIGNIFICANCE_LEVEL",
    "THESIS_VIEW",
    "Column",
    "DistributionRow",
    "distribution_cell_values",
    "distribution_groups",
    "render_distribution_note",
    "render_distribution_table",
    "render_scenario_note",
    "render_scenario_table",
    "scenario_cells",
]

#: The view the thesis reports: (b), absolute-normalised.
THESIS_VIEW = "absolute"
#: The threshold the marker stands for.
SIGNIFICANCE_LEVEL = 0.05

_VIEW_TEXT: Mapping[str, str] = {
    "absolute": (
        "the absolute-normalised view (b): distances saturate at {distance} km"
    ),
    "raw": "the RAW view (a): distances are normalised by the grid diagonal",
}


@dataclass(frozen=True)
class Column:
    """One column of a scenario table."""

    key: str
    header: str
    plain: str
    #: ``"higher"`` or ``"lower"`` is better; ``None`` marks no best value.
    better: str | None
    decimals: int


SCENARIO_COLUMNS: tuple[Column, ...] = (
    Column("land_use", "Land Use", "Land Use", "higher", 3),
    Column("woodland", "Woodland", "Woodland", "higher", 3),
    Column("urban", "Urban", "Urban", "higher", 3),
    Column("c_align", "$C_{\\text{align}}$", "C_align", "lower", 3),
    Column("total_conversions", "Conversions", "Conversions", None, 1),
)


@dataclass(frozen=True)
class DistributionRow:
    """One row of the distribution-accuracy table."""

    key: str
    label: str
    plain: str
    decimals: int


DISTRIBUTION_ROWS: tuple[DistributionRow, ...] = (
    DistributionRow(
        "skill_climatology",
        "CRPS skill vs climatology",
        "CRPS skill vs climatology",
        3,
    ),
    DistributionRow(
        "skill_initialisation",
        "CRPS skill vs initialisation",
        "CRPS skill vs initialisation",
        3,
    ),
    DistributionRow("ks_statistic", "PIT KS statistic", "PIT KS statistic", 3),
    DistributionRow(
        "coverage_50",
        "Coverage of the central 50\\,\\% interval",
        "Coverage of the central 50 % interval",
        3,
    ),
    DistributionRow(
        "coverage_90",
        "Coverage of the central 90\\,\\% interval",
        "Coverage of the central 90 % interval",
        3,
    ),
    DistributionRow("mae", "MAE of $\\mathrm{E}[Z]$", "MAE of E[Z]", 2),
    DistributionRow(
        "within_iteration_sd",
        "Quality-score SD within an iteration",
        "Quality-score SD within an iteration",
        4,
    ),
    DistributionRow(
        "degenerate_share",
        "Stakeholders approving all or none",
        "Share of stakeholders approving all or none",
        3,
    ),
    DistributionRow("approval_rate", "Greedy approval rate", "Greedy approval rate", 3),
)


# ---------------------------------------------------------------- scenario tables
def scenario_cells(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every (configuration, scenario) block of a campaign summary, in order."""
    return [
        cell
        for configuration in summary.get("configurations", [])
        for cell in configuration.get("scenarios", [])
    ]


def _entry(block: Mapping[str, Any], view: str, column: Column) -> Any:
    if column.key == "total_conversions":
        return block.get("total_conversions")
    return (block.get(view) or {}).get(column.key)


def _test(
    cell: Mapping[str, Any], view: str, method: str, column: Column
) -> Mapping[str, Any] | None:
    by_view = cell.get("tests_by_view")
    if not isinstance(by_view, Mapping) or view not in by_view:
        raise ValueError(
            "summary.json has no paired tests for the "
            f"{view!r} view; rebuild it with "
            "`python -m chapter4_qolu.run_scenarios --tag <tag> --aggregate`"
        )
    return (by_view[view].get(method) or {}).get(column.key)


def _is_marked(test: Mapping[str, Any] | None) -> bool:
    p_value = None if test is None else test.get("p_value")
    return p_value is not None and float(p_value) < SIGNIFICANCE_LEVEL


def _best(
    cell: Mapping[str, Any], view: str, methods: Sequence[str], column: Column
) -> set[str]:
    """Methods holding the best mean at full precision (ties all bold)."""
    if column.better is None:
        return set()
    means = {
        method: float(entry["mean"])
        for method in methods
        if (entry := _entry(cell["methods"][method], view, column)) is not None
    }
    if not means:
        return set()
    target = max(means.values()) if column.better == "higher" else min(means.values())
    return {method for method, value in means.items() if value == target}


def _latex_cell(entry: Any, decimals: int, bold: bool, marked: bool) -> str:
    if entry is None:
        return "--"
    body = (
        f"{float(entry['mean']):.{decimals}f} \\pm "
        f"{float(entry.get('se', 0.0)):.{decimals}f}"
    )
    if bold:
        body = f"\\mathbf{{{body}}}"
    if marked:
        body += "^{*}"
    return f"${body}$"


def _seed_counts(cell: Mapping[str, Any], methods: Sequence[str]) -> list[int]:
    return sorted({int(cell["methods"][method]["n_seeds"]) for method in methods})


def _seed_text(counts: Sequence[int]) -> str:
    if len(counts) == 1:
        return f"{counts[0]} seeds"
    return f"{counts[0]}--{counts[-1]} seeds"


def _view_text(summary: Mapping[str, Any], view: str) -> str:
    distance = summary.get("absolute_max_distance_km")
    shown = "?" if distance is None else f"{float(distance):g}"
    return _VIEW_TEXT[view].format(distance=shown)


def _weights_text(cell: Mapping[str, Any]) -> str:
    weights = cell.get("weights") or {}
    return ", ".join(
        f"{name.replace('_', ' ')} {float(weights[name]):.2f}"
        for name in ("land_use", "woodland", "urban")
        if name in weights
    )


def render_scenario_table(
    summary: Mapping[str, Any],
    cell: Mapping[str, Any],
    view: str = THESIS_VIEW,
    reference: str = "qolu_buia",
) -> str:
    """One scenario's results as a booktabs ``table`` environment."""
    if view not in _VIEW_TEXT:
        raise ValueError(f"unknown view {view!r}; expected one of {tuple(_VIEW_TEXT)}")
    methods = ordered_methods(cell["methods"])
    best = {
        column.key: _best(cell, view, methods, column) for column in SCENARIO_COLUMNS
    }
    scenario = int(cell["scenario"])
    lines = [
        (
            f"% Chapter 4, Scenario {scenario} - generated by "
            "chapter4_qolu/make_thesis_tables.py; see the .txt note beside it."
        ),
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{_scenario_caption(summary, cell, view, methods)}}}",
        f"  \\label{{tab:p2-results-s{scenario}}}",
        "  \\begin{tabular}{l" + "c" * len(SCENARIO_COLUMNS) + "}",
        "    \\toprule",
        "    Method & "
        + " & ".join(column.header for column in SCENARIO_COLUMNS)
        + " \\\\",
        "    \\midrule",
    ]
    for method in methods:
        cells = [METHOD_LABELS.get(method, method)]
        for column in SCENARIO_COLUMNS:
            entry = _entry(cell["methods"][method], view, column)
            marked = method != reference and _is_marked(
                _test(cell, view, method, column)
            )
            cells.append(
                _latex_cell(entry, column.decimals, method in best[column.key], marked)
            )
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _scenario_caption(
    summary: Mapping[str, Any],
    cell: Mapping[str, Any],
    view: str,
    methods: Sequence[str],
) -> str:
    weights = _weights_text(cell)
    view_text = _view_text(summary, view).replace(" km", "~km")
    return (
        f"Scenario {int(cell['scenario'])}"
        + (f" (weights: {weights})" if weights else "")
        + f". Mean $\\pm$ standard error over {_seed_text(_seed_counts(cell, methods))}. "
        f"Land Use, Woodland and Urban are scored in {view_text}; higher is "
        "better. $C_{\\text{align}}$ is the mean total-variation distance "
        "between the neighbourhood land-type distributions before and after a "
        "run; lower is better. Conversions counts the parcels converted. Bold "
        "marks the best value in a column. $^{*}$ marks $p < 0.05$ in a "
        "two-sided paired $t$-test against QOLU+BUIA over seeds."
    )


def _p_text(test: Mapping[str, Any] | None) -> str:
    if test is None:
        return "no test"
    if not test.get("possible", True) or test.get("p_value") is None:
        return str(test.get("note") or "no test possible")
    p_value = float(test["p_value"])
    t_value = test.get("t")
    t_text = (
        "inf"
        if t_value is not None and math.isinf(float(t_value))
        else ("n/a" if t_value is None else f"{float(t_value):+.3f}")
    )
    return (
        f"mean difference {float(test['mean_difference']):+.4g}, t = {t_text}, "
        f"df = {test.get('df')}, p = {p_value:.3g}"
    )


def _parcel_count(configuration: Mapping[str, Any]) -> int | None:
    rows = configuration.get("grid_height") or configuration.get("grid_size")
    columns = configuration.get("grid_width") or configuration.get("grid_size")
    return int(rows) * int(columns) if rows and columns else None


def _c_align_lines(
    cell: Mapping[str, Any],
    methods: Sequence[str],
    view: str,
    configuration: Mapping[str, Any] | None,
) -> list[str]:
    """The observed ratio of ``C_align x |P|`` to the conversion count, per run.

    Design note: when a conversion turns a whole non-urban parcel urban, the
    neighbourhood distribution of each adjacent parcel ``j`` moves by exactly
    ``1 / n_j`` whatever the land types were, so ``C_align`` sums to about one
    ``1 / |P|`` per conversion.  The note states the observed range so that a
    lower ``C_align`` is not read as more than the conversion count shows.
    """
    if configuration is None:
        return []
    parcels = _parcel_count(configuration)
    if not parcels:
        return []
    ratios = [
        value * parcels / count
        for method in methods
        for value, count in zip(
            (cell["methods"][method].get(view) or {})
            .get("c_align", {})
            .get("values", []),
            cell["methods"][method].get("total_conversions", {}).get("values", []),
        )
        if count
    ]
    if not ratios:
        return []
    lines = [
        (
            f"C_align x |P| / Conversions, per run in this table (|P| = {parcels} "
            f"parcels): {min(ratios):.3f} to {max(ratios):.3f}."
        )
    ]
    if float(configuration.get("conversion_fraction", 0.0)) == 1.0:
        lines.append(
            "  Every conversion here turns a whole non-urban parcel urban, which "
            "moves the land-type distribution of each adjacent parcel j by "
            "exactly 1/n_j whatever the land types were, so C_align is close to "
            "Conversions / |P| and differences in C_align follow differences in "
            "the number of conversions."
        )
    return lines


def render_scenario_note(
    summary: Mapping[str, Any],
    cell: Mapping[str, Any],
    view: str = THESIS_VIEW,
    reference: str = "qolu_buia",
    source: str = "summary.json",
    configuration: Mapping[str, Any] | None = None,
) -> str:
    """The plain-text note that says what every number of one table is.

    ``configuration`` (the summary block the cell belongs to) adds the
    observed ``C_align``-to-conversions ratio; without it the line is omitted.
    """
    methods = ordered_methods(cell["methods"])
    counts = _seed_counts(cell, methods)
    seeds = sorted(
        {int(seed) for method in methods for seed in cell["methods"][method]["seeds"]}
    )
    scenario = int(cell["scenario"])
    lines = [
        f"Table: Scenario {scenario} results (table_scenario{scenario}.tex)",
        "",
        (
            f"Source: {source}, campaign "
            f"'{summary.get('campaign', {}).get('tag', '?')}', "
            f"{summary.get('n_runs', '?')} run records in total."
        ),
        f"Scenario weights: {_weights_text(cell) or 'not recorded'}.",
        "Rows (method keys): "
        + ", ".join(f"{METHOD_LABELS.get(m, m)} ({m})" for m in methods)
        + ".",
        "",
        (
            "Each cell: mean +- standard error (sample SD with ddof = 1, divided by "
            f"sqrt(n)) over n = {_seed_text(counts)} ({', '.join(map(str, seeds))}), "
            "one run per seed of "
            f"{cell.get('iterations', '?')} iterations."
        ),
        (
            f"Land Use, Woodland, Urban: {_view_text(summary, view)}. Higher is "
            "better. Woodland has no distance term, so it is the same in the RAW and "
            "absolute views."
        ),
        (
            "C_align: mean over every parcel of the total-variation distance "
            "between the land-type distribution of its (up to eight) adjacent "
            "parcels before and after the run; lower is better; the same in every "
            "view."
        ),
        (
            "Conversions: parcels converted in the run; one decimal; no best value "
            "is marked."
        ),
        *_c_align_lines(cell, methods, view, configuration),
        (
            "Bold: the best mean at full precision in the column (highest for Land "
            "Use, Woodland and Urban; lowest for C_align); none for Conversions."
        ),
        (
            f"*: two-sided paired t-test of QOLU+BUIA ({reference}) against the "
            "row's method, paired by seed, on the column's quantity in the view "
            f"shown, p < {SIGNIFICANCE_LEVEL:g}; df = n - 1.  The p-values are not "
            "corrected for multiple comparisons."
        ),
        "",
        "Paired tests (QOLU+BUIA minus the row's method):",
    ]
    for method in methods:
        if method == reference:
            continue
        for column in SCENARIO_COLUMNS:
            lines.append(
                f"  {METHOD_LABELS.get(method, method)}, {column.plain}: "
                f"{_p_text(_test(cell, view, method, column))}"
            )
    derived = (summary.get("derived_metrics") or {}).get("c_align")
    if derived:
        lines.extend(
            [
                "",
                (
                    "C_align provenance across the campaign: "
                    f"{derived.get('recorded', 0)} records computed it during the run, "
                    f"{derived.get('from_stored_grids', 0)} were scored from their "
                    f"stored initial and final grids, {derived.get('missing', 0)} "
                    "lack it."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------- the distribution-accuracy table
def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def distribution_cell_values(cell: Mapping[str, Any], target: str) -> dict[str, float]:
    """One diagnostics cell's value of every distribution-accuracy row.

    The distribution statistics are the cell's pooled values against
    ``target``; the two policy quantities and the share are averaged over the
    cell's stakeholders first, so every cell weighs the same in the
    across-cell mean.
    """
    statistics = cell["targets"][target]
    stakeholders = (cell.get("policy") or {}).get("stakeholders") or []
    if not stakeholders:
        raise ValueError(
            f"cell {cell.get('tag', '?')} has no policy block; re-run or "
            "re-render it with the current diagnostics driver"
        )
    values = {
        key: float(statistics[key])
        for key in (
            "skill_climatology",
            "skill_initialisation",
            "ks_statistic",
            "coverage_50",
            "coverage_90",
            "mae",
        )
    }
    values["within_iteration_sd"] = _mean(
        [float(s["quality_score"]["within_iteration_sd"]) for s in stakeholders]
    )
    values["approval_rate"] = _mean([float(s["approval_rate"]) for s in stakeholders])
    values["degenerate_share"] = _mean(
        [1.0 if s["degenerate_approval"] else 0.0 for s in stakeholders]
    )
    return values


def distribution_groups(
    sweep: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], str]:
    """Columns (scenarios, then pooled), their per-row summaries and the target."""
    target = str(sweep.get("headline_target", "tail_return"))
    cells = list(sweep.get("cells") or [])
    if not cells:
        raise ValueError("the diagnostics sweep has no finished cells")
    per_cell = [
        (int(cell["scenario"]), distribution_cell_values(cell, target))
        for cell in cells
    ]
    scenarios = sorted({scenario for scenario, _ in per_cell})
    groups: dict[str, list[dict[str, float]]] = {
        f"Scenario {scenario}": [v for s, v in per_cell if s == scenario]
        for scenario in scenarios
    }
    groups["Pooled"] = [values for _, values in per_cell]
    summaries = {
        name: {
            row.key: summarise([values[row.key] for values in members])
            for row in DISTRIBUTION_ROWS
        }
        | {"n": len(members)}
        for name, members in groups.items()
    }
    return list(groups), summaries, target


def render_distribution_table(sweep: Mapping[str, Any]) -> str:
    """The distribution-accuracy table as a booktabs ``table`` environment."""
    names, summaries, target = distribution_groups(sweep)
    counts = sorted({int(summaries[name]["n"]) for name in names[:-1]})
    caption = (
        "Accuracy of the QOLU return distribution and use of the quality score "
        f"in a greedy evaluation rollout. Mean $\\pm$ standard error over seeds "
        f"({'/'.join(map(str, counts))} per scenario; "
        f"{summaries['Pooled']['n']} pooled), scored against the "
        f"{target.replace('_', ' ')}."
    )
    lines = [
        (
            "% Chapter 4, distribution accuracy - generated by "
            "chapter4_qolu/make_thesis_tables.py; see table_distribution.txt."
        ),
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:p2-distribution}",
        "  \\begin{tabular}{l" + "c" * len(names) + "}",
        "    \\toprule",
        "    Statistic & " + " & ".join(names) + " \\\\",
        "    \\midrule",
    ]
    for row in DISTRIBUTION_ROWS:
        cells = [row.label]
        for name in names:
            sample = summaries[name][row.key]
            cells.append(
                f"${sample.mean:.{row.decimals}f} \\pm "
                f"{sample.se:.{row.decimals}f}$"
            )
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_distribution_note(
    sweep: Mapping[str, Any],
    source: str = "sweep",
    learner: tuple[int, Sequence[int]] | None = None,
) -> str:
    """The plain-text note for the distribution-accuracy table.

    ``learner`` is ``(t_n, syncs per stakeholder)`` as the cells' own learner
    blocks report it; the sweep's settings record ``null`` for "the package
    default", which says where the value came from but not what it was.
    """
    names, summaries, target = distribution_groups(sweep)
    settings = sweep.get("settings") or {}
    lines = [
        "Table: return-distribution accuracy (table_distribution.tex)",
        "",
        (
            f"Source: {source}; tag '{sweep.get('tag', '?')}', "
            f"{sweep.get('n_complete', '?')} of {sweep.get('n_cells', '?')} cells "
            "finished."
        ),
        (
            f"Run: method {settings.get('method', '?')}, window "
            f"{settings.get('window', '?')}, configuration "
            f"{settings.get('configuration', '?')}, reward reading "
            f"{settings.get('qolu_reward_mode', '?')}, "
            f"{settings.get('stakeholders', '?')} stakeholders, "
            f"{settings.get('training_iterations', '?')} training iterations, "
            f"scenarios {settings.get('scenarios', '?')}, seeds "
            f"{settings.get('seeds', '?')}."
        ),
        (
            "Evaluation: a greedy rollout of "
            f"{settings.get('evaluation_iterations', '?')} iterations after "
            "training (epsilon 0, NoisyNet noise zeroed, no learning)."
        ),
        "",
        "Each cell: mean +- standard error (ddof = 1) over diagnostics cells, "
        "one cell = one (scenario, seed) run; n = "
        + ", ".join(f"{name}: {summaries[name]['n']}" for name in names)
        + ".",
        (
            f"Distribution rows are scored against the {target} reading of the "
            "realised return (the bootstrap-free discounted tail of the appraisal "
            "stream), pooled over the cell's stakeholders:"
        ),
        (
            "  CRPS skill vs climatology: 1 - CRPS(model) / CRPS(state-blind "
            "marginal histogram of the cell's returns); positive = better than "
            "climatology."
        ),
        (
            "  CRPS skill vs initialisation: 1 - CRPS(model) / CRPS(the same "
            "network at initialisation)."
        ),
        (
            "  PIT KS statistic: Kolmogorov-Smirnov distance of the randomised PIT "
            "from U(0, 1); an effect size (its p-value is not reported because the "
            "appraisals are not independent)."
        ),
        (
            "  Coverage of the central 50 % / 90 % interval: share of realised "
            "returns inside the predicted central interval."
        ),
        "  MAE of E[Z]: mean absolute error of the predicted mean return.",
        (
            "Policy rows come from the same rollout and are averaged over the "
            "cell's stakeholders before the across-cell mean:"
        ),
        (
            "  Quality-score SD within an iteration: SD of Q(s, approve) - "
            "Q(s, reject) across the candidates of one decision buffer, averaged "
            "over iterations."
        ),
        (
            "  Stakeholders approving all or none: share of the cell's stakeholders "
            "whose greedy approval rate over the rollout is exactly 0 or 1."
        ),
        "  Greedy approval rate: share of appraised candidates approved.",
        "",
    ]
    if learner is not None:
        interval, syncs = learner
        counts = "/".join(str(value) for value in sorted(set(syncs)))
        lines.extend(
            [
                (
                    f"Learner: t_n = {interval} gradient steps "
                    f"(target_sync_interval), {counts} target-network syncs "
                    "per stakeholder in every cell."
                ),
                "",
            ]
        )
    lines.append("Values at full precision (mean, SE):")
    for row in DISTRIBUTION_ROWS:
        lines.append(
            f"  {row.plain}: "
            + "; ".join(
                f"{name} {summaries[name][row.key].mean:.6g} "
                f"({summaries[name][row.key].se:.3g})"
                for name in names
            )
        )
    return "\n".join(lines) + "\n"
