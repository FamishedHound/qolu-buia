"""Tests for the markdown / LaTeX result tables."""

from __future__ import annotations

from typing import Any

import pytest

from chapter4_qolu.reporting.aggregate import aggregate_records
from chapter4_qolu.reporting.tables import (
    METHOD_LABELS,
    PAPER_ROW_ORDER,
    format_cell,
    ordered_methods,
    render_latex,
    render_markdown,
)
from chapter4_qolu.tests.test_reporting_aggregate import make_record

_SERIES = {
    # qolu - baseline is a *varying* difference so the t-tests are finite.
    "qolu_buia": [0.80, 0.86, 0.83],
    "pmaddpg": [0.70, 0.71, 0.75],
    "ltos": [0.60, 0.65, 0.61],
    "mitm": [0.50, 0.58, 0.52],
}


@pytest.fixture()
def summary() -> dict[str, Any]:
    records = [
        make_record(
            method,
            seed,
            {"land_use": value, "woodland": value, "urban": value},
            scenario=scenario,
        )
        for scenario in (1, 2)
        for method, series in _SERIES.items()
        for seed, value in enumerate(series)
    ]
    return aggregate_records(records, metadata={"tag": "unit", "seeds": [0, 1, 2]})


class TestRowOrder:
    def test_follows_the_published_table_order(self) -> None:
        assert ordered_methods({"qolu_buia", "mitm", "pmaddpg", "ltos"}) == (
            "ltos",
            "pmaddpg",
            "mitm",
            "qolu_buia",
        )

    def test_keeps_variants_next_to_their_parent_method(self) -> None:
        order = ordered_methods(set(PAPER_ROW_ORDER))
        assert order.index("fixed_ltos") == order.index("ltos") + 1
        assert order.index("mitm_learned") == order.index("mitm") + 1
        # QOLU has variants of its own, and they follow their parent like
        # every other variant does.
        assert order.index("qolu_c51") == order.index("qolu_buia") + 1
        assert order.index("qolu_buia_per") == order.index("qolu_buia") + 2

    def test_the_published_system_is_the_last_of_the_published_rows(self) -> None:
        order = ordered_methods(set(PAPER_ROW_ORDER))
        published = [
            method
            for method in order
            if method in ("ltos", "pmaddpg", "mitm", "qolu_buia")
        ]
        assert published[-1] == "qolu_buia"

    def test_unknown_methods_are_appended_alphabetically(self) -> None:
        order = ordered_methods({"qolu_buia", "zeta", "alpha"})
        assert order == ("qolu_buia", "alpha", "zeta")


class TestCellFormatting:
    def test_plain_cell_is_mean_plus_minus_standard_error(self) -> None:
        assert format_cell(0.8213, 0.0125, style="markdown") == "0.821 ± 0.013"

    def test_significance_marker_is_a_star(self) -> None:
        cell = format_cell(0.5, 0.01, significant=True, style="markdown")
        assert cell.endswith("*")

    def test_best_cell_is_bold(self) -> None:
        assert format_cell(0.5, 0.01, best=True, style="markdown").startswith("**")
        assert format_cell(0.5, 0.01, best=True, style="latex").startswith("\\textbf{")

    def test_latex_uses_the_maths_plus_minus(self) -> None:
        assert "$\\pm$" in format_cell(0.5, 0.01, style="latex")

    def test_missing_values_render_as_a_dash(self) -> None:
        assert format_cell(None, None, style="markdown") == "-"


class TestMarkdownTables:
    def test_has_a_table_per_configuration_and_scenario_in_every_view(
        self, summary: dict[str, Any]
    ) -> None:
        text = render_markdown(summary)
        # one configuration x two scenarios x three views
        assert text.count("| Method | Land Use | Woodland | Urban |") == 6

    def test_main_section_is_the_min_max_view(self, summary: dict[str, Any]) -> None:
        text = render_markdown(summary)
        main = text.index("view (c)")
        appendix_raw = text.index("view (a)")
        appendix_absolute = text.index("view (b)")
        assert main < appendix_raw < appendix_absolute

    def test_every_caption_names_its_view(self, summary: dict[str, Any]) -> None:
        text = render_markdown(summary)
        captions = [line for line in text.splitlines() if line.startswith("*Table")]
        assert len(captions) == 6
        assert all(
            ("view (a)" in caption)
            or ("view (b)" in caption)
            or ("view (c)" in caption)
            for caption in captions
        )

    def test_rows_use_the_published_labels_in_the_published_order(
        self, summary: dict[str, Any]
    ) -> None:
        text = render_markdown(summary)
        rows = [line for line in text.splitlines() if line.startswith("| LToS")]
        assert rows
        block = text[text.index("| LToS") :]
        positions = [
            block.index(f"| {METHOD_LABELS[method]} |")
            for method in ("ltos", "pmaddpg", "mitm", "qolu_buia")
        ]
        assert positions == sorted(positions)

    def test_marks_the_best_method_of_each_column_in_bold(
        self, summary: dict[str, Any]
    ) -> None:
        text = render_markdown(summary)
        qolu_row = next(
            line for line in text.splitlines() if line.startswith("| QOLU+BUIA ")
        )
        assert qolu_row.count("**") == 6  # three bolded columns

    def test_stars_a_baseline_the_reference_beats_significantly(
        self, summary: dict[str, Any]
    ) -> None:
        text = render_markdown(summary)
        mitm_row = next(
            line for line in text.splitlines() if line.startswith("| MitM ")
        )
        assert "*" in mitm_row

    def test_states_how_the_comparison_is_made(self, summary: dict[str, Any]) -> None:
        text = render_markdown(summary)
        assert "every configuration is listed" in text
        assert "pattern_check.md" in text

    def test_names_the_campaign_tag(self, summary: dict[str, Any]) -> None:
        assert "unit" in render_markdown(summary)


class TestLatexTables:
    def test_emits_one_table_environment_per_cell_and_view(
        self, summary: dict[str, Any]
    ) -> None:
        text = render_latex(summary)
        assert text.count("\\begin{table}") == 6
        assert text.count("\\end{table}") == 6

    def test_uses_the_published_four_column_layout(
        self, summary: dict[str, Any]
    ) -> None:
        text = render_latex(summary)
        assert "\\begin{tabular}{|c|c|c|c|}" in text
        assert (
            "\\textbf{Method} & \\textbf{Land Use} & \\textbf{Woodland} & "
            "\\textbf{Urban}" in text
        )

    def test_labels_are_unique_and_latex_safe(self, summary: dict[str, Any]) -> None:
        text = render_latex(summary)
        labels = [
            line.strip()
            for line in text.splitlines()
            if line.strip().startswith("\\la")
        ]
        assert len(labels) == 6
        assert len(set(labels)) == 6
        assert all("_" not in label for label in labels)

    def test_escapes_underscores_in_captions(self, summary: dict[str, Any]) -> None:
        text = render_latex(summary)
        for line in text.splitlines():
            if line.strip().startswith("\\caption"):
                assert "_" not in line.replace("\\_", "")
