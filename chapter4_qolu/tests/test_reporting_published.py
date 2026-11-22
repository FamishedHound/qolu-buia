"""Tests for the published Tables 4-7 constant and the closeness statistic.

The constant is transcribed once from
``thesis/overleaf/chapters/chapter4_paper2.tex``; the first test parses that
file and asserts the transcription is exact, so the constant cannot drift away
from the chapter it claims to quote.  The chapter lives outside this package
(and is read-only), so the test skips when it is not there.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from chapter4_qolu.reporting.published import (
    PUBLISHED_CLOSENESS_VIEW,
    PUBLISHED_METHOD_ORDER,
    PUBLISHED_TABLES,
    normalisation_check,
    published_means,
    value_closeness,
)

_CHAPTER = (
    Path(__file__).resolve().parents[2]
    / "thesis"
    / "overleaf"
    / "chapters"
    / "chapter4_paper2.tex"
)
_ROW = re.compile(
    r"(LToS|P-MADDPG|MitM|\\textbf\{QOLU\+BUIA\})\s*&\s*"
    r"([\d.]+)\s*\$\\pm\$\s*([\d.]+)\s*&\s*"
    r"([\d.]+)\s*\$\\pm\$\s*([\d.]+)\s*&\s*"
    r"([\d.]+)\s*\$\\pm\$\s*([\d.]+)"
)
_TEX_LABEL = {
    "LToS": "ltos",
    "P-MADDPG": "pmaddpg",
    "MitM": "mitm",
    "\\textbf{QOLU+BUIA}": "qolu_buia",
}
_TABLE_LABELS = (
    "tab:p2-results-uniform",
    "tab:p2-results-scenario2",
    "tab:p2-results-scenario3",
    "tab:p2-results-scenario4",
)


def _parse_chapter() -> dict[int, dict[str, dict[str, tuple[float, float]]]]:
    text = _CHAPTER.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(r"\\begin\{table\}.*?\\end\{table\}", text, re.DOTALL)
    parsed: dict[int, dict[str, dict[str, tuple[float, float]]]] = {}
    for scenario, label in enumerate(_TABLE_LABELS, start=1):
        block = next(item for item in blocks if label in item)
        rows: dict[str, dict[str, tuple[float, float]]] = {}
        for match in _ROW.finditer(block):
            method = _TEX_LABEL[match.group(1)]
            numbers = [float(value) for value in match.groups()[1:]]
            rows[method] = {
                "land_use": (numbers[0], numbers[1]),
                "woodland": (numbers[2], numbers[3]),
                "urban": (numbers[4], numbers[5]),
            }
        parsed[scenario] = rows
    return parsed


class TestTranscription:
    def test_the_constant_matches_the_chapter_tex(self) -> None:
        if not _CHAPTER.is_file():
            pytest.skip(f"chapter source not available: {_CHAPTER}")
        parsed = _parse_chapter()
        assert set(parsed) == set(PUBLISHED_TABLES)
        for scenario, methods in parsed.items():
            assert set(methods) == set(PUBLISHED_TABLES[scenario])
            for method, metrics in methods.items():
                for metric, (mean, se) in metrics.items():
                    cell = PUBLISHED_TABLES[scenario][method][metric]
                    assert cell.mean == pytest.approx(mean)
                    assert cell.se == pytest.approx(se)

    def test_there_are_forty_eight_cells(self) -> None:
        cells = [
            cell
            for methods in PUBLISHED_TABLES.values()
            for metrics in methods.values()
            for cell in metrics.values()
        ]
        assert len(cells) == 48

    def test_every_scenario_has_the_four_published_methods(self) -> None:
        for methods in PUBLISHED_TABLES.values():
            assert tuple(sorted(methods)) == tuple(sorted(PUBLISHED_METHOD_ORDER))

    def test_published_means_is_metric_major(self) -> None:
        means = published_means(1)
        assert means["land_use"]["qolu_buia"] == pytest.approx(0.80)
        assert means["urban"]["mitm"] == pytest.approx(0.70)


class TestNormalisationCheck:
    def test_no_published_column_contains_both_a_zero_and_a_one(self) -> None:
        check = normalisation_check()
        assert check["minmax_compatible"] is False
        assert check["n_columns"] == 12
        assert check["n_columns_with_zero_and_one"] == 0

    def test_the_check_reports_the_observed_range(self) -> None:
        check = normalisation_check()
        assert check["minimum"] == pytest.approx(0.60)
        assert check["maximum"] == pytest.approx(0.90)

    def test_the_closeness_view_is_not_the_minmax_one(self) -> None:
        # The published cells never touch 0 or 1, so the cross-method min-max
        # view cannot be what they are; the closeness statistic therefore uses
        # the view whose scale does match.
        assert PUBLISHED_CLOSENESS_VIEW == "absolute"


class TestValueCloseness:
    def test_an_exact_copy_of_the_published_table_scores_zero(self) -> None:
        values = published_means(1)
        assert value_closeness(values, scenario=1) == pytest.approx(0.0)

    def test_a_constant_offset_is_the_mean_absolute_difference(self) -> None:
        values = {
            metric: {method: mean + 0.1 for method, mean in per_method.items()}
            for metric, per_method in published_means(2).items()
        }
        assert value_closeness(values, scenario=2) == pytest.approx(0.1)

    def test_missing_methods_are_skipped_not_counted_as_zero(self) -> None:
        values = published_means(3)
        trimmed = {
            metric: {"qolu_buia": per_method["qolu_buia"] + 0.2}
            for metric, per_method in values.items()
        }
        assert value_closeness(trimmed, scenario=3) == pytest.approx(0.2)

    def test_an_empty_cell_has_no_closeness(self) -> None:
        assert value_closeness({}, scenario=1) is None
