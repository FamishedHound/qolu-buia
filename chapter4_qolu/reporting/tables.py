"""Markdown and LaTeX result tables in the layout of the paper's Tables 4-7.

The published tables have four rows (LToS, P-MADDPG, MitM and "US", i.e.
QOLU + BUIA) and three columns (Land Use, Woodland, Urban), with each cell a
mean and a standard error.  That layout is reproduced here for every
(configuration, scenario) cell of a campaign and for each of the three views
:mod:`chapter4_qolu.reporting.aggregate` carries.

Design note (row order): the published order is LToS, P-MADDPG, MitM, US.  The
variants this package implements (``fixed_ltos``, ``mitm_learned`` and the
others) are placed immediately after their parent method, so the published four
keep their published order whenever only they are present.  The
``no_regulator`` ablation, which has no published row, comes first.

Design note (significance marker): the ``*`` marks a *baseline* whose paired
t-test against QOLU + BUIA has ``p < 0.05`` on the view of the table it is in:
the RAW tables mark the RAW test and the absolute tables the absolute test
(``tests_by_view`` in ``summary.json``).  The min-max tables mark the RAW test,
because min-max rescaling is an affine map per cell and cannot change a paired
test.

Design note (columns): the RAW and absolute tables carry the total-variation
conversion metric ``C_align`` as a fourth column when the records have it
(lower is better, so its bold marks the lowest); the min-max tables keep the
published three.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from chapter4_qolu.reporting.aggregate import LOWER_IS_BETTER, PUBLISHED_METRICS

__all__ = [
    "DEFAULT_DECIMALS",
    "METHOD_LABELS",
    "METRIC_LABELS",
    "PAPER_ROW_ORDER",
    "VIEW_TITLES",
    "format_cell",
    "ordered_methods",
    "render_latex",
    "render_markdown",
]

#: Row order of the published tables, with this package's extra variants placed
#: immediately after the method they vary.
PAPER_ROW_ORDER: tuple[str, ...] = (
    "no_regulator",
    "ltos",
    "fixed_ltos",
    "pmaddpg",
    "pmaddpg_v1",
    "pmaddpg_reg_only",
    "mitm",
    "mitm_learned",
    "qolu_buia",
    "qolu_c51",
    "qolu_buia_per",
)

METHOD_LABELS: Mapping[str, str] = {
    "no_regulator": "No regulator",
    "ltos": "LToS",
    "fixed_ltos": "fixed LToS",
    "pmaddpg": "P-MADDPG",
    "pmaddpg_v1": "P-MADDPG (v1)",
    "pmaddpg_reg_only": "P-MADDPG (reg only)",
    "mitm": "MitM",
    "mitm_learned": "MitM (learned)",
    "qolu_buia": "QOLU+BUIA",
    "qolu_c51": "QOLU+BUIA (C51, no noisy net)",
    "qolu_buia_per": "QOLU+BUIA (prioritised replay)",
}

METRIC_LABELS: Mapping[str, str] = {
    "land_use": "Land Use",
    "woodland": "Woodland",
    "urban": "Urban",
    "c_align": "C_align",
}

#: Column headers of ``tables.tex``.
_LATEX_HEADERS: Mapping[str, str] = {
    "land_use": "\\textbf{Land Use}",
    "woodland": "\\textbf{Woodland}",
    "urban": "\\textbf{Urban}",
    "c_align": "\\textbf{$C_{\\mathrm{align}}$}",
}
#: Extra columns of the RAW and absolute tables, shown when the records carry them.
_EXTRA_COLUMNS: tuple[str, ...] = ("c_align",)
#: The view whose paired test a table of each view marks.
_TEST_VIEW: Mapping[str, str] = {"raw": "raw", "absolute": "absolute", "minmax": "raw"}

#: ``(letter, heading, caption fragment)`` per view, in reporting order.
VIEW_TITLES: Mapping[str, tuple[str, str, str]] = {
    "minmax": (
        "C",
        "view (c): cross-method min-max normalisation",
        (
            "view (c) cross-method min-max: within each column the worst "
            "method scores 0 and the best 1, rescaled from the RAW view"
        ),
    ),
    "raw": (
        "A",
        "view (a): RAW metric values",
        (
            "view (a) RAW metric values exactly as "
            "chapter4_qolu/env/evaluation.py defines them, with distances "
            "normalised by the grid diagonal"
        ),
    ),
    "absolute": (
        "B",
        "view (b): absolute-normalised metric values",
        (
            "view (b) absolute-normalised: the same metrics with "
            "RewardConfig.max_distance set to a physical distance, so "
            "proximity saturates at a fixed number of kilometres rather than "
            "at the grid diagonal"
        ),
    ),
}

_VIEW_ORDER: tuple[str, ...] = ("minmax", "raw", "absolute")
#: Design note: three decimals.  The between-seed spread of these metrics is of
#: order 1e-3, so two decimals would round it away.  Four are available through
#: ``--decimals``.
DEFAULT_DECIMALS: int = 3


def ordered_methods(methods: Iterable[str]) -> tuple[str, ...]:
    """Sort method keys into the published row order."""
    present = set(methods)
    known = tuple(method for method in PAPER_ROW_ORDER if method in present)
    unknown = tuple(sorted(present - set(PAPER_ROW_ORDER)))
    return known + unknown


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def format_cell(
    mean: float | None,
    se: float | None,
    *,
    best: bool = False,
    significant: bool = False,
    style: str = "markdown",
    decimals: int = DEFAULT_DECIMALS,
) -> str:
    """One ``mean +- SE`` table cell."""
    if mean is None:
        return "-"
    spread = 0.0 if se is None else float(se)
    separator = "$\\pm$" if style == "latex" else "±"
    body = f"{float(mean):.{decimals}f} {separator} {spread:.{decimals}f}"
    if significant:
        body += "*"
    if not best:
        return body
    return f"\\textbf{{{body}}}" if style == "latex" else f"**{body}**"


def _cell_values(
    cell: Mapping[str, Any], view: str, method: str, metric: str
) -> tuple[float | None, float | None]:
    block = cell["methods"].get(method, {}).get(view, {})
    entry = block.get(metric)
    if entry is None:
        return None, None
    return float(entry["mean"]), float(entry.get("se", 0.0))


def _columns(cell: Mapping[str, Any], view: str) -> tuple[str, ...]:
    """The metric columns of one table: the published three, plus extras."""
    if view == "minmax":
        return PUBLISHED_METRICS
    extras = tuple(
        metric
        for metric in _EXTRA_COLUMNS
        if any(metric in (block.get(view) or {}) for block in cell["methods"].values())
    )
    return (*PUBLISHED_METRICS, *extras)


def _best_methods(
    cell: Mapping[str, Any],
    view: str,
    methods: Sequence[str],
    columns: Sequence[str] = PUBLISHED_METRICS,
) -> dict[str, str | None]:
    best: dict[str, str | None] = {}
    for metric in columns:
        scored = {
            method: value
            for method in methods
            if (value := _cell_values(cell, view, method, metric)[0]) is not None
        }
        if not scored:
            best[metric] = None
        elif metric in LOWER_IS_BETTER:
            best[metric] = min(scored, key=lambda name: (scored[name], name))
        else:
            best[metric] = max(scored, key=lambda name: (scored[name], name))
    return best


def _tests_for(cell: Mapping[str, Any], view: str) -> Mapping[str, Any]:
    """The paired tests a table of ``view`` marks (see the module note)."""
    test_view = _TEST_VIEW.get(view, "raw")
    by_view = cell.get("tests_by_view")
    if isinstance(by_view, Mapping) and test_view in by_view:
        return by_view[test_view]
    # A summary without per-view tests: only RAW tests exist.
    return cell.get("tests", {}) if test_view == "raw" else {}


def _is_significant(
    cell: Mapping[str, Any], method: str, metric: str, view: str = "raw"
) -> bool:
    entry = _tests_for(cell, view).get(method, {}).get(metric)
    return bool(entry and entry.get("significant"))


def _rows(
    cell: Mapping[str, Any],
    view: str,
    methods: Sequence[str],
    style: str,
    decimals: int = DEFAULT_DECIMALS,
) -> list[list[str]]:
    columns = _columns(cell, view)
    best = _best_methods(cell, view, methods, columns)
    rows = []
    for method in methods:
        row = [_label(method)]
        for metric in columns:
            mean, se = _cell_values(cell, view, method, metric)
            row.append(
                format_cell(
                    mean,
                    se,
                    best=best[metric] == method,
                    significant=_is_significant(cell, method, metric, view),
                    style=style,
                    decimals=decimals,
                )
            )
        rows.append(row)
    return rows


def _weights_text(weights: Mapping[str, Any]) -> str:
    if not weights:
        return ""
    parts = [
        f"{METRIC_LABELS[name].lower()} {float(weights[name]):.2f}"
        for name in PUBLISHED_METRICS
        if name in weights
    ]
    return ", ".join(parts)


def _seed_count(cell: Mapping[str, Any]) -> int:
    counts = {block["n_seeds"] for block in cell["methods"].values()}
    return max(counts) if counts else 0


def _caption(
    summary: Mapping[str, Any],
    configuration: Mapping[str, Any],
    cell: Mapping[str, Any],
    view: str,
    style: str = "markdown",
) -> str:
    _, _, fragment = VIEW_TITLES[view]
    if view == "absolute":
        distance = summary.get("absolute_max_distance_km")
        if distance is not None:
            fragment += f" (max_distance = {float(distance):.1f} km)"
    latex = style == "latex"
    spread = "mean $\\pm$ SE" if latex else "mean ± SE"
    threshold = "$p < 0.05$" if latex else "p < 0.05"
    description = configuration.get("description")
    scale = f"{configuration['label']}" + (f" ({description})" if description else "")
    tested_on = {
        "raw": "on this RAW view",
        "absolute": "on this absolute-normalised view",
        "minmax": (
            "on the RAW view, of which this view is a per-cell affine "
            "rescaling that leaves the test unchanged"
        ),
    }[view]
    extra = (
        "; C_align is lower-is-better, so its bold marks the lowest"
        if "c_align" in _columns(cell, view)
        else ""
    )
    return (
        f"Scenario {cell['scenario']}, configuration {scale}: {fragment}. Cells "
        f"are {spread} over {_seed_count(cell)} seed(s); * marks {threshold} in "
        f"a paired t-test against QOLU+BUIA {tested_on}; bold marks the "
        f"best method in the column{extra}."
    )


def _cells(
    summary: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    return [
        (configuration, cell)
        for configuration in summary["configurations"]
        for cell in configuration["scenarios"]
    ]


def _preamble(summary: Mapping[str, Any]) -> list[str]:
    campaign = summary.get("campaign", {})
    tag = campaign.get("tag", "campaign")
    lines = [f"# Campaign `{tag}` - result tables", ""]
    details = []
    for key in ("generated", "raster", "iterations", "methods", "scenarios", "seeds"):
        if key in campaign:
            details.append(f"* **{key}**: {campaign[key]}")
    details.append(f"* **runs aggregated**: {summary.get('n_runs', 0)}")
    lines.extend(details)
    lines.extend(
        [
            "",
            (
                "Three readings of the same five metrics are reported and "
                "never silently mixed:"
            ),
            "",
            f"* **(a) RAW** - {summary['views']['raw']}",
            f"* **(b) absolute-normalised** - {summary['views']['absolute']}",
            f"* **(c) cross-method min-max** - {summary['views']['minmax']}",
            "",
            (
                "Cells are `mean ± SE` over the campaign's seeds. **Bold** "
                "marks the best method in a column. `*` marks a baseline whose "
                "paired t-test against QOLU+BUIA (paired over seeds) has "
                "p < 0.05 on the view of the table it appears in; the min-max "
                "tables use the RAW-view test, which a per-cell affine "
                "rescaling leaves unchanged.  The RAW and absolute tables add "
                "the corrected conversion metric C_align (lower is better) "
                "when the records carry it."
            ),
            "",
            (
                "Configurations are compared with the published ordering; "
                "every configuration is listed. See `pattern_check.md` for the "
                "ranking comparison against the published Tables 4-7."
            ),
            "",
        ]
    )
    return lines


def render_markdown(
    summary: Mapping[str, Any], decimals: int = DEFAULT_DECIMALS
) -> str:
    """The full ``tables.md`` document."""
    lines = _preamble(summary)
    for view in _VIEW_ORDER:
        letter, heading, _ = VIEW_TITLES[view]
        prefix = "## Main tables - " if view == "minmax" else f"## Appendix {letter} - "
        lines.extend([f"{prefix}{heading}", ""])
        for index, (configuration, cell) in enumerate(_cells(summary), start=1):
            methods = ordered_methods(cell["methods"])
            weights = _weights_text(cell.get("weights", {}))
            lines.append(
                f"### Configuration `{configuration['label']}`, Scenario "
                f"{cell['scenario']}" + (f" ({weights})" if weights else "")
            )
            lines.append("")
            columns = _columns(cell, view)
            lines.append(
                "| Method | "
                + " | ".join(METRIC_LABELS[metric] for metric in columns)
                + " |"
            )
            lines.append("| --- |" + " --- |" * len(columns))
            for row in _rows(cell, view, methods, "markdown", decimals):
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
            lines.append(
                f"*Table {letter}{index} - "
                f"{_caption(summary, configuration, cell, view)}*"
            )
            lines.append("")
    return "\n".join(lines)


def _escape_latex(text: str) -> str:
    """Escape the characters a caption can plausibly contain."""
    for character in ("_", "%", "&", "#"):
        text = text.replace(character, f"\\{character}")
    return text


def _latex_label(
    summary: Mapping[str, Any], label: str, scenario: int, view: str
) -> str:
    tag = str(summary.get("campaign", {}).get("tag", "campaign"))
    cleaned = f"{tag}-{label}".replace("_", "-").replace(".", "p")
    return f"tab:c4-{cleaned}-s{scenario}-{VIEW_TITLES[view][0].lower()}"


def render_latex(summary: Mapping[str, Any], decimals: int = DEFAULT_DECIMALS) -> str:
    """The full ``tables.tex`` fragment (``\\input``-able, no preamble)."""
    tag = summary.get("campaign", {}).get("tag", "campaign")
    lines = [
        f"% Chapter 4 campaign '{tag}' - generated by chapter4_qolu/run_scenarios.py.",
        "% An includable fragment: no document class, no preamble.",
        "% Views: (c) cross-method min-max, (a) RAW, (b) absolute-normalised.",
        "",
    ]
    for view in _VIEW_ORDER:
        lines.append(f"% ---- {VIEW_TITLES[view][1]} ----")
        for configuration, cell in _cells(summary):
            methods = ordered_methods(cell["methods"])
            caption = _escape_latex(
                _caption(summary, configuration, cell, view, style="latex")
            )
            columns = _columns(cell, view)
            header = (
                "\\textbf{Method} & "
                + " & ".join(_LATEX_HEADERS[metric] for metric in columns)
                + " \\\\ \\hline"
            )
            lines.extend(
                [
                    "\\begin{table}[!ht]",
                    "\\centering",
                    "\\resizebox{\\linewidth}{!}{",
                    "\\begin{tabular}{|" + "c|" * (len(columns) + 1) + "}",
                    "\\hline",
                    header,
                ]
            )
            for row in _rows(cell, view, methods, "latex", decimals):
                lines.append(" & ".join(row) + " \\\\ \\hline")
            label = _latex_label(
                summary, str(configuration["label"]), int(cell["scenario"]), view
            )
            lines.extend(
                [
                    "\\end{tabular}",
                    "}",
                    f"\\caption{{{caption}}}",
                    f"\\label{{{label}}}",
                    "\\end{table}",
                    "",
                ]
            )
        lines.append("")
    return "\n".join(lines)
