"""Run records published without their land-cover grids.

The records distributed under ``results/`` leave ``initial_grid`` and
``final_grid`` out: they are per-parcel land-cover classes derived from the
UKCEH Land Cover Map 2015, which may not be redistributed.  The commands that
need the grids must stop and say how to regenerate them; the ones that do not
must work as before.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chapter4_qolu import make_figures, run_scenarios
from chapter4_qolu.reporting.aggregate import aggregate_records
from chapter4_qolu.reporting.derived import (
    LCM2015_DOI,
    MISSING,
    has_stored_grids,
    metric_source,
    missing_grids_message,
)
from chapter4_qolu.reporting.documents import write_campaign_documents
from chapter4_qolu.reporting.paths import run_json_path
from chapter4_qolu.tests.test_figures import METHODS, SCALE, _record

TAG = "published"


def _campaign(root: Path, record_c_align: bool) -> Path:
    """A campaign written with its grids, then stripped of them as published."""
    directory = root / f"campaign_{TAG}"
    records = []
    for scenario in (1, 2):
        for seed in (0, 1):
            for index, method in enumerate(METHODS):
                record = _record(method, scenario, seed, offset=0.02 * index)
                if record_c_align:
                    for view in ("final_metrics", "final_metrics_absolute"):
                        record[view]["c_align"] = 0.1 + 0.01 * index
                path = run_json_path(directory, SCALE, scenario, method, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record), encoding="utf-8")
                records.append(record)
    summary = aggregate_records(records, metadata={"tag": TAG})
    write_campaign_documents(directory, summary, records=records)
    for path in directory.glob("*/scenario*/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        del record["initial_grid"], record["final_grid"]
        path.write_text(json.dumps(record), encoding="utf-8")
    return directory


@pytest.fixture()
def published(tmp_path: Path) -> Path:
    return _campaign(tmp_path, record_c_align=False)


def _a_record(campaign: Path) -> dict:
    path = next(campaign.glob("*/scenario*/*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


class TestTheMessage:
    def test_it_names_the_product_and_the_way_back(self) -> None:
        text = missing_grids_message("Drawing X", "somewhere", 3)
        assert text.startswith("Drawing X needs the land-cover grids")
        assert "3 record(s) under somewhere carry none" in text
        assert "UKCEH Land Cover Map 2015" in text
        assert LCM2015_DOI in text
        assert "rerun the experiments" in text and "without --resume" in text

    def test_a_published_record_has_no_grids(self, published: Path) -> None:
        record = _a_record(published)
        assert not has_stored_grids(record)
        assert metric_source(record) == MISSING


class TestTheFigures:
    def test_the_land_cover_figures_stop_with_the_way_back(
        self, published: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "figures"
        with pytest.raises(SystemExit) as stopped:
            make_figures.main(["--campaign", str(published), "--out", str(out)])
        message = str(stopped.value)
        assert message.startswith("error: Drawing the land-cover figures")
        assert LCM2015_DOI in message and "--no-land-cover" in message
        assert not out.exists()

    def test_the_region_figures_stop_too(self, published: Path, tmp_path: Path) -> None:
        out = tmp_path / "figures"
        with pytest.raises(SystemExit):
            make_figures.main(
                ["--campaign", str(published), "--region", "--out", str(out)]
            )
        assert not out.exists()

    def test_without_land_cover_the_metric_figures_are_drawn(
        self, published: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "figures"
        arguments = ["--campaign", str(published), "--region", "--no-land-cover"]
        assert make_figures.main([*arguments, "--out", str(out)]) == 0
        names = {path.stem for path in out.glob("*.csv")}
        assert f"scenario_comparison_{TAG}" in names
        assert "curves_scenario1" in names
        assert "table_bars_scenario1_absolute" in names
        assert not any(
            name.startswith(("land_cover_", "region_", "land_preference"))
            for name in names
        )
        index = (out / "figures_index.md").read_text(encoding="utf-8")
        assert "land_cover_scenario" not in index and "region_" not in index


class TestTheAggregation:
    def test_it_stops_when_c_align_needs_the_grids(self, published: Path) -> None:
        summary = published / "summary.json"
        before = summary.read_bytes()
        arguments = ["--tag", TAG, "--aggregate", "--output-dir", str(published.parent)]
        with pytest.raises(SystemExit) as stopped:
            run_scenarios.main(arguments)
        message = str(stopped.value)
        assert message.startswith("error: Rebuilding the campaign documents")
        assert "c_align" in message and LCM2015_DOI in message
        assert summary.read_bytes() == before

    def test_records_that_recorded_c_align_still_aggregate(
        self, tmp_path: Path
    ) -> None:
        campaign = _campaign(tmp_path, record_c_align=True)
        arguments = ["--tag", TAG, "--aggregate", "--output-dir", str(tmp_path)]
        assert run_scenarios.main(arguments) == 0
        summary = json.loads((campaign / "summary.json").read_text(encoding="utf-8"))
        assert summary["derived_metrics"]["c_align"]["missing"] == 0
        assert summary["n_runs"] == 16
