"""Paired tests per view, so a table's ``*`` always refers to the view it shows.

The thesis reports view (b), absolute-normalised; its tables must mark the
tests computed on view (b), and the RAW tables the tests computed on RAW.  The
summary stores both, with ``t``, ``p`` and the mean paired difference per cell,
and the total-variation ``c_align`` and the conversion count are tested beside the
three published metrics.  ``c_align`` is lower-is-better, which every ordering
- best in bold, the ranking, the min-max rescaling - has to respect.
"""

from __future__ import annotations

from typing import Any

import pytest

from chapter4_qolu.reporting.aggregate import (
    LOWER_IS_BETTER,
    PUBLISHED_METRICS,
    aggregate_records,
    min_max_normalise,
    rank_methods,
)
from chapter4_qolu.reporting.tables import render_latex, render_markdown
from chapter4_qolu.tests.test_reporting_aggregate import make_record

#: Land Use differs between the methods in the absolute view only.
RAW_LAND_USE = {"qolu_buia": [0.80, 0.86, 0.83], "mitm": [0.81, 0.85, 0.84]}
ABSOLUTE_LAND_USE = {"qolu_buia": [0.90, 0.95, 0.92], "mitm": [0.50, 0.58, 0.52]}
C_ALIGN = {"qolu_buia": [0.10, 0.12, 0.11], "mitm": [0.30, 0.33, 0.31]}


def records() -> list[dict[str, Any]]:
    out = []
    for method in ("qolu_buia", "mitm"):
        for seed in range(3):
            record = make_record(
                method,
                seed,
                {
                    "land_use": RAW_LAND_USE[method][seed],
                    "c_align": C_ALIGN[method][seed],
                },
                absolute={"land_use": ABSOLUTE_LAND_USE[method][seed]},
            )
            record["total_conversions"] = (
                (780 + seed) if method == "qolu_buia" else 1000
            )
            out.append(record)
    return out


@pytest.fixture()
def cell() -> dict[str, Any]:
    return aggregate_records(records())["configurations"][0]["scenarios"][0]


@pytest.fixture()
def summary() -> dict[str, Any]:
    return aggregate_records(records(), metadata={"tag": "views"})


class TestTheSummaryStoresATestPerView:
    def test_both_views_are_tested(self, cell: dict[str, Any]) -> None:
        assert set(cell["tests_by_view"]) == {"raw", "absolute"}

    def test_each_view_uses_its_own_values(self, cell: dict[str, Any]) -> None:
        raw = cell["tests_by_view"]["raw"]["mitm"]["land_use"]
        absolute = cell["tests_by_view"]["absolute"]["mitm"]["land_use"]
        assert raw["p_value"] > 0.05
        assert absolute["p_value"] < 0.05
        assert absolute["mean_difference"] == pytest.approx(
            sum(
                a - b
                for a, b in zip(
                    ABSOLUTE_LAND_USE["qolu_buia"], ABSOLUTE_LAND_USE["mitm"]
                )
            )
            / 3
        )

    def test_t_p_and_the_mean_difference_are_stored(self, cell: dict[str, Any]) -> None:
        entry = cell["tests_by_view"]["absolute"]["mitm"]["land_use"]
        assert {"t", "p_value", "mean_difference", "df", "n"} <= set(entry)
        assert entry["df"] == 2

    def test_c_align_and_conversions_are_tested_too(self, cell: dict[str, Any]) -> None:
        tested = cell["tests_by_view"]["absolute"]["mitm"]
        assert set(tested) == {*PUBLISHED_METRICS, "c_align", "total_conversions"}
        assert tested["c_align"]["mean_difference"] < 0.0
        assert tested["total_conversions"]["mean_difference"] == pytest.approx(-219.0)

    def test_the_legacy_block_is_unchanged(self, cell: dict[str, Any]) -> None:
        assert set(cell["tests"]["mitm"]) == set(PUBLISHED_METRICS)
        assert cell["tests"]["mitm"] == {
            metric: cell["tests_by_view"]["raw"]["mitm"][metric]
            for metric in PUBLISHED_METRICS
        }


class TestLowerIsBetterForCAlign:
    def test_c_align_is_the_lower_is_better_metric(self) -> None:
        assert "c_align" in LOWER_IS_BETTER

    def test_the_ranking_puts_the_lowest_first(self, cell: dict[str, Any]) -> None:
        assert cell["ranking"]["c_align"] == ["qolu_buia", "mitm"]
        assert rank_methods({"a": 0.3, "b": 0.1}, lower_is_better=True) == ("b", "a")

    def test_min_max_scores_the_lowest_one(self, cell: dict[str, Any]) -> None:
        assert cell["methods"]["qolu_buia"]["minmax"]["c_align"]["mean"] == 1.0
        assert cell["methods"]["mitm"]["minmax"]["c_align"]["mean"] == 0.0
        assert min_max_normalise({"a": 0.3, "b": 0.1}, lower_is_better=True) == {
            "a": 0.0,
            "b": 1.0,
        }


def _starred(cell: str) -> bool:
    """Whether a markdown cell carries the significance marker (bold aside)."""
    value = cell.strip()
    if value.startswith("**") and value.endswith("**"):
        value = value[2:-2]
    return value.endswith("*")


def _row(text: str, heading: str, label: str) -> str:
    section = text[text.index(heading) :]
    return next(line for line in section.splitlines() if line.startswith(f"| {label} "))


class TestTheTablesMarkTheViewTheyShow:
    def test_the_absolute_table_marks_the_absolute_test(
        self, summary: dict[str, Any]
    ) -> None:
        text = render_markdown(summary)
        absolute = _row(text, "view (b)", "MitM")
        raw = _row(text, "view (a)", "MitM")
        assert _starred(absolute.split("|")[2])
        assert not _starred(raw.split("|")[2])

    def test_captions_name_the_view_the_test_was_run_on(
        self, summary: dict[str, Any]
    ) -> None:
        captions = [
            line
            for line in render_markdown(summary).splitlines()
            if line.startswith("*Table")
        ]
        absolute = [caption for caption in captions if "view (b)" in caption]
        assert absolute and all("absolute-normalised view" in c for c in absolute)
        assert all("on the RAW view" not in c for c in absolute)

    def test_raw_and_absolute_tables_carry_a_c_align_column(
        self, summary: dict[str, Any]
    ) -> None:
        text = render_markdown(summary)
        assert text.count("| Method | Land Use | Woodland | Urban | C_align |") == 2
        assert text.count("| Method | Land Use | Woodland | Urban |\n") == 1

    def test_the_lowest_c_align_is_bold(self, summary: dict[str, Any]) -> None:
        row = _row(render_markdown(summary), "view (b)", "QOLU+BUIA")
        assert row.split("|")[5].strip().startswith("**")

    def test_the_latex_tables_follow_the_same_rule(
        self, summary: dict[str, Any]
    ) -> None:
        text = render_latex(summary)
        assert "\\textbf{$C_{\\mathrm{align}}$}" in text
