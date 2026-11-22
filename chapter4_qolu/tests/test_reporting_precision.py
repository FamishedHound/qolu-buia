"""Tests for the finer-precision reporting: per-seed values and per_seed.csv.

The published Tables 4-7 quote two decimals on a 0.05 lattice; the chapter
has to be able to report more than that, so ``summary.json`` carries the
raw per-seed vectors and every campaign writes a full-precision ``per_seed.csv``
next to its documents.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from chapter4_qolu.reporting.aggregate import ALL_METRICS, aggregate_records
from chapter4_qolu.reporting.documents import DOCUMENT_NAMES, write_campaign_documents
from chapter4_qolu.reporting.per_seed import PER_SEED_COLUMNS, render_per_seed_csv
from chapter4_qolu.reporting.tables import render_latex, render_markdown
from chapter4_qolu.tests.test_reporting_aggregate import cell, make_record

_SEED_VALUES = (0.123456789, 0.234567891, 0.345678912)


def _records() -> list[dict]:
    return [
        make_record("qolu_buia", seed, {"land_use": value})
        for seed, value in enumerate(_SEED_VALUES)
    ] + [
        make_record("pmaddpg", seed, {"land_use": value / 2.0})
        for seed, value in enumerate(_SEED_VALUES)
    ]


class TestPerSeedValuesInSummary:
    def test_raw_view_carries_every_seed_value(self) -> None:
        summary = aggregate_records(_records())
        block = cell(summary)["methods"]["qolu_buia"]["raw"]["land_use"]
        assert block["values"] == pytest.approx(list(_SEED_VALUES))
        assert block["n"] == 3

    def test_values_are_ordered_by_seed(self) -> None:
        summary = aggregate_records(_records())
        block = cell(summary)["methods"]["qolu_buia"]
        assert block["seeds"] == [0, 1, 2]
        assert block["raw"]["land_use"]["values"][0] == pytest.approx(_SEED_VALUES[0])

    def test_absolute_and_minmax_views_carry_values_too(self) -> None:
        summary = aggregate_records(_records())
        block = cell(summary)["methods"]["qolu_buia"]
        assert "values" in block["absolute"]["land_use"]
        assert "values" in block["minmax"]["land_use"]

    def test_values_keep_full_double_precision(self) -> None:
        summary = aggregate_records(_records())
        block = cell(summary)["methods"]["qolu_buia"]["raw"]["land_use"]
        restored = json.loads(json.dumps(block))["values"]
        assert restored == pytest.approx(list(_SEED_VALUES), rel=0, abs=0)

    def test_mean_can_be_recomputed_from_the_values(self) -> None:
        summary = aggregate_records(_records())
        block = cell(summary)["methods"]["qolu_buia"]["raw"]["land_use"]
        assert sum(block["values"]) / len(block["values"]) == pytest.approx(
            block["mean"]
        )


#: Mean of ``_SEED_VALUES``; the RAW ``land_use`` cell of QOLU + BUIA.
_SEED_MEAN = sum(_SEED_VALUES) / len(_SEED_VALUES)


class TestTableDecimals:
    def test_markdown_defaults_to_three_decimals(self) -> None:
        text = render_markdown(aggregate_records(_records()))
        assert f"{_SEED_MEAN:.3f}" in text
        assert f"{_SEED_MEAN:.4f}" not in text

    def test_markdown_accepts_four_decimals(self) -> None:
        text = render_markdown(aggregate_records(_records()), decimals=4)
        assert f"{_SEED_MEAN:.4f}" in text

    def test_latex_accepts_four_decimals(self) -> None:
        text = render_latex(aggregate_records(_records()), decimals=4)
        assert f"{_SEED_MEAN:.4f}" in text


class TestPerSeedCsv:
    def test_columns_are_the_documented_ones(self) -> None:
        assert PER_SEED_COLUMNS[:5] == (
            "configuration",
            "scenario",
            "method",
            "seed",
            "window",
        )
        for metric in ALL_METRICS:
            assert metric in PER_SEED_COLUMNS
        for name in ("total_conversions", "saturation_iteration", "runtime_seconds"):
            assert name in PER_SEED_COLUMNS

    def test_one_row_per_run(self) -> None:
        rows = list(csv.DictReader(render_per_seed_csv(_records()).splitlines()))
        assert len(rows) == 6
        assert {row["method"] for row in rows} == {"qolu_buia", "pmaddpg"}

    def test_metric_values_keep_full_precision(self) -> None:
        rows = list(csv.DictReader(render_per_seed_csv(_records()).splitlines()))
        values = sorted(
            float(row["land_use"]) for row in rows if row["method"] == "qolu_buia"
        )
        assert values == pytest.approx(list(_SEED_VALUES), rel=0, abs=0)
        assert "0.123456789" in render_per_seed_csv(_records())

    def test_records_the_window_when_one_was_pinned(self) -> None:
        records = _records()
        for record in records:
            record["window"] = {"label": "E495_N140"}
        rows = list(csv.DictReader(render_per_seed_csv(records).splitlines()))
        assert {row["window"] for row in rows} == {"E495_N140"}

    def test_rows_are_sorted_deterministically(self) -> None:
        rows = list(csv.DictReader(render_per_seed_csv(_records()).splitlines()))
        keys = [(row["scenario"], row["method"], row["seed"]) for row in rows]
        assert keys == sorted(keys)

    def test_written_alongside_the_campaign_documents(self, tmp_path: Path) -> None:
        summary = aggregate_records(_records())
        written = write_campaign_documents(tmp_path, summary, records=_records())
        assert "per_seed.csv" in DOCUMENT_NAMES
        assert written["per_seed.csv"].is_file()
        text = written["per_seed.csv"].read_text(encoding="utf-8")
        assert text.splitlines()[0].startswith("configuration,scenario,method,seed")
