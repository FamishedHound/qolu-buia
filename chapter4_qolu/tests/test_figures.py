"""Figure readiness: tiny synthetic campaigns, no long runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.figures import backend
from chapter4_qolu.figures.bars import table_bars
from chapter4_qolu.figures.curves import iteration_curves
from chapter4_qolu.figures.data import grids_by_method, load_campaign
from chapter4_qolu.figures.index import render_figures_index
from chapter4_qolu.figures.maps import land_cover_maps
from chapter4_qolu.figures.palette import (
    AGGREGATE_CLASS_COLOURS,
    NO_DATA_COLOUR,
    read_colour_table,
)
from chapter4_qolu.figures.scan_map import scan_score_map
from chapter4_qolu.make_figures import main
from chapter4_qolu.reporting.aggregate import aggregate_records
from chapter4_qolu.reporting.documents import write_campaign_documents
from chapter4_qolu.reporting.paths import parse_scale_config, run_json_path
from chapter4_qolu.run_record import encode_class_grid

METHODS = ("qolu_buia", "pmaddpg", "ltos", "mitm")
SCALE = parse_scale_config("6:1.0:10")


def _record(method: str, scenario: int, seed: int, offset: float) -> dict:
    rng = np.random.default_rng(abs(hash((method, scenario, seed))) % 2**32)
    initial = rng.integers(0, 10, size=(6, 6)).astype(np.int64)
    final = initial.copy()
    final[: 2 + seed, :] = 9
    iterations = 8
    return {
        "method": method,
        "method_name": method,
        "scenario": scenario,
        "seed": seed,
        "iterations": iterations,
        "weights": {"land_use": 0.33, "woodland": 0.33, "urban": 0.33},
        "configuration": SCALE.as_record(),
        "window": {"label": "E495_N140", "raster_origin": [0, 0], "pinned": True},
        "raster": {"path": "synthetic.img", "placeholder_region": False},
        "absolute_max_distance_km": 10.0,
        "total_conversions": 10 + seed,
        "saturation_iteration": None,
        "runtime_seconds": 1.0,
        "final_metrics": {
            "land_use": 0.3 + offset,
            "woodland": 0.4 + offset,
            "urban": 0.5 + offset,
            "conversion_alignment": 0.1,
            "well_being": 0.2,
        },
        "final_metrics_absolute": {
            "land_use": 0.6 + offset,
            "woodland": 0.7 + offset,
            "urban": 0.8 + offset,
            "conversion_alignment": 0.1,
            "well_being": 0.2,
        },
        "initial_grid": encode_class_grid(initial),
        "final_grid": encode_class_grid(final),
        "per_iteration": {
            "team_reward": [float(i) * 0.1 + offset for i in range(iterations)],
            "urban_fraction": [0.1 * i for i in range(iterations)],
            "n_conversions": [1] * iterations,
            "n_proposals": [2] * iterations,
            "n_approved": [1] * iterations,
            "n_candidates": [5] * iterations,
            "conversion_alignment": [0.05 * i for i in range(iterations)],
            "well_being": [0.2] * iterations,
        },
    }


@pytest.fixture()
def campaign(tmp_path: Path) -> Path:
    directory = tmp_path / "campaign_demo"
    records = []
    for scenario in (1, 2):
        for seed in (0, 1):
            for index, method in enumerate(METHODS):
                record = _record(method, scenario, seed, offset=0.02 * index)
                path = run_json_path(directory, SCALE, scenario, method, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record), encoding="utf-8")
                records.append(record)
    summary = aggregate_records(records, metadata={"tag": "demo"})
    write_campaign_documents(directory, summary, records=records)
    return directory


@pytest.fixture()
def scan_summary(tmp_path: Path) -> Path:
    payload = {
        "tag": "demo_scan",
        "grid": {
            "bounding_box_km": {
                "east_min": 420,
                "east_max": 640,
                "north_min": 90,
                "north_max": 250,
            },
            "window_size_km": 50,
            "stride_km": 25,
            "max_sea_fraction": 0.3,
            "eastings": [420, 445],
            "northings": [90, 115],
            "n_candidates": 4,
            "n_retained": 3,
        },
        "stage1": {
            "windows": [
                {
                    "window": {
                        "label": "E420_N90",
                        "easting_km": 420,
                        "northing_km": 90,
                        "size_km": 50,
                        "description": "E 420-470 km, N 90-140 km",
                    },
                    "composition": {"sea_fraction": 0.1, "class_shares": {"3": 0.5}},
                    "score": {
                        "first_place_count": 1,
                        "kendall_sum": 0.3,
                        "reference_minmax_mean": 0.5,
                        "flat_metrics": 0,
                        "joint_first_count": 1,
                        "metrics": [],
                    },
                },
                {
                    "window": {
                        "label": "E445_N115",
                        "easting_km": 445,
                        "northing_km": 115,
                        "size_km": 50,
                        "description": "E 445-495 km, N 115-165 km",
                    },
                    "composition": {"sea_fraction": 0.0, "class_shares": {"4": 0.6}},
                    "score": {
                        "first_place_count": 2,
                        "kendall_sum": 0.7,
                        "reference_minmax_mean": 0.8,
                        "flat_metrics": 0,
                        "joint_first_count": 2,
                        "metrics": [],
                    },
                },
            ]
        },
        "stage2": {
            "windows": [
                {
                    "window": {"label": "E445_N115", "description": "x"},
                    "matched_cells": 5,
                    "total_cells": 12,
                }
            ]
        },
        "selection": {"winner": {"label": "E445_N115"}},
        "dropped": [],
    }
    path = tmp_path / "scan_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestPalette:
    def test_every_aggregate_class_has_a_colour(self) -> None:
        assert len(AGGREGATE_CLASS_COLOURS) == 10
        for colour in AGGREGATE_CLASS_COLOURS.values():
            assert colour.startswith("#") and len(colour) == 7

    def test_no_data_has_its_own_colour(self) -> None:
        assert NO_DATA_COLOUR not in set(AGGREGATE_CLASS_COLOURS.values())

    def test_a_clr_file_is_read_in_the_products_order(self, tmp_path: Path) -> None:
        path = tmp_path / "raster.img.clr"
        path.write_text("0 240 240 240\n1 255 0 0\n2 0 102 0\n", encoding="utf-8")
        table = read_colour_table(path)
        assert table[0] == "#f0f0f0"
        assert table[1] == "#ff0000"
        assert table[2] == "#006600"

    def test_the_builtin_palette_matches_the_shipped_clr_values(self) -> None:
        # The .clr codes are class numbers; the palette is keyed by band index.
        assert AGGREGATE_CLASS_COLOURS[0] == "#ff0000"
        assert AGGREGATE_CLASS_COLOURS[9] == "#808080"


class TestData:
    def test_a_campaign_loads_its_records_and_summary(self, campaign: Path) -> None:
        loaded = load_campaign(campaign)
        assert len(loaded.records) == 16
        assert loaded.summary["n_runs"] == 16

    def test_grids_are_keyed_by_method(self, campaign: Path) -> None:
        loaded = load_campaign(campaign)
        grids = grids_by_method(
            [r for r in loaded.records if r["scenario"] == 1 and r["seed"] == 0]
        )
        assert set(grids) == set(METHODS)
        initial, final = grids["mitm"]
        assert initial.shape == (6, 6)
        assert final.shape == (6, 6)


class TestFigures:
    def test_land_cover_maps_are_written(self, campaign: Path, tmp_path: Path) -> None:
        loaded = load_campaign(campaign)
        out = tmp_path / "figures"
        result = land_cover_maps(loaded, out, scenario=1, seed=0)
        assert result.name == "land_cover_scenario1_seed0"
        for path in result.paths:
            assert path.is_file()
        assert any(path.suffix == ".csv" for path in result.paths)

    def test_curves_are_written_per_scenario(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        loaded = load_campaign(campaign)
        out = tmp_path / "figures"
        result = iteration_curves(loaded, out, scenario=2)
        assert result.name == "curves_scenario2"
        assert all(path.is_file() for path in result.paths)

    def test_bars_are_written_per_scenario_and_view(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        loaded = load_campaign(campaign)
        out = tmp_path / "figures"
        result = table_bars(loaded, out, scenario=1, view="minmax")
        assert result.name == "table_bars_scenario1_minmax"
        assert all(path.is_file() for path in result.paths)

    def test_the_scan_map_is_written(self, scan_summary: Path, tmp_path: Path) -> None:
        payload = json.loads(scan_summary.read_text(encoding="utf-8"))
        out = tmp_path / "figures"
        result = scan_score_map(payload, out)
        assert result.name == "scan_map_demo_scan"
        assert all(path.is_file() for path in result.paths)

    def test_pdf_and_png_are_produced_when_matplotlib_is_available(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        if not backend.matplotlib_available():
            pytest.skip("matplotlib is not importable in this interpreter")
        loaded = load_campaign(campaign)
        result = table_bars(loaded, tmp_path / "figures", scenario=1, view="raw")
        suffixes = {path.suffix for path in result.paths}
        assert {".pdf", ".png"} <= suffixes

    def test_without_matplotlib_only_the_data_file_is_written(
        self, campaign: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(backend, "matplotlib_available", lambda: False)
        loaded = load_campaign(campaign)
        result = table_bars(loaded, tmp_path / "figures", scenario=1, view="raw")
        assert {path.suffix for path in result.paths} == {".csv"}


class TestIndex:
    def test_the_index_lists_every_figure_with_its_source(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        loaded = load_campaign(campaign)
        out = tmp_path / "figures"
        results = [
            land_cover_maps(loaded, out, scenario=1, seed=0),
            table_bars(loaded, out, scenario=1, view="minmax"),
        ]
        text = render_figures_index(results)
        for result in results:
            assert result.name in text
            assert result.source in text
            assert result.command in text


class TestCli:
    def test_the_cli_builds_every_campaign_figure(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "figures"
        assert main(["--campaign", str(campaign), "--out", str(out)]) == 0
        assert (out / "figures_index.md").is_file()
        names = {path.stem for path in out.glob("*.csv")}
        assert "land_cover_scenario1_seed0" in names
        assert "curves_scenario1" in names
        assert "table_bars_scenario1_minmax" in names

    def test_the_cli_builds_a_scan_map(
        self, scan_summary: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "figures"
        assert main(["--scan", str(scan_summary.parent), "--out", str(out)]) == 0
        assert (out / "scan_map_demo_scan.csv").is_file()

    def test_file_names_are_deterministic(self, campaign: Path, tmp_path: Path) -> None:
        first = tmp_path / "a"
        second = tmp_path / "b"
        main(["--campaign", str(campaign), "--out", str(first)])
        main(["--campaign", str(campaign), "--out", str(second)])
        assert sorted(p.name for p in first.iterdir()) == sorted(
            p.name for p in second.iterdir()
        )
