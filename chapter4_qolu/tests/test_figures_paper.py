"""The three published-figure families: scenario comparison, pies, region maps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.figures import backend
from chapter4_qolu.figures.data import load_campaign
from chapter4_qolu.figures.land_preference import (
    LEGEND_NAME,
    land_preference_figures,
    land_preference_table,
    paper_class_grid,
)
from chapter4_qolu.figures.paper_classes import (
    PAPER_CLASSES,
    aggregate_to_paper,
    paper_class_colours,
    paper_class_names,
)
from chapter4_qolu.figures.region_maps import region_before_after, region_converted
from chapter4_qolu.figures.scenario_comparison import (
    PUBLISHED_METRIC_LEGEND,
    PUBLISHED_X_ORDER,
    scenario_comparison,
)
from chapter4_qolu.land_cover import AggregateClass, band_index
from chapter4_qolu.make_figures import main
from chapter4_qolu.reporting.aggregate import aggregate_records
from chapter4_qolu.reporting.documents import write_campaign_documents
from chapter4_qolu.reporting.paths import parse_scale_config, run_json_path
from chapter4_qolu.run_record import encode_class_grid

METHODS = ("qolu_buia", "pmaddpg", "ltos", "mitm")
SCALE = parse_scale_config("8:1.0:10")


def _record(method: str, scenario: int, seed: int, offset: float) -> dict:
    rng = np.random.default_rng(abs(hash((method, scenario, seed))) % 2**32)
    initial = rng.integers(0, 10, size=(8, 8)).astype(np.int64)
    initial[0, 0] = -1  # one sea cell
    final = initial.copy()
    final[1:3, :] = band_index(AggregateClass.BUILT_UP)
    return {
        "method": method,
        "scenario": scenario,
        "seed": seed,
        "iterations": 5,
        "weights": {"land_use": 0.33, "woodland": 0.33, "urban": 0.33},
        "configuration": SCALE.as_record(),
        "window": {"label": "SE_region", "raster_origin": [0, 0], "pinned": True},
        "raster": {"path": "synthetic.img", "placeholder_region": False},
        "absolute_max_distance_km": 10.0,
        "total_conversions": 16,
        "saturation_iteration": None,
        "runtime_seconds": 1.0,
        "final_metrics": {
            "land_use": 0.30 + offset,
            "woodland": 0.40 + offset,
            "urban": 0.50 + offset,
            "conversion_alignment": 0.1,
            "well_being": 0.2,
        },
        "final_metrics_absolute": {
            "land_use": 0.60 + offset,
            "woodland": 0.70 + offset,
            "urban": 0.80 + offset,
            "conversion_alignment": 0.1,
            "well_being": 0.2,
        },
        "initial_grid": encode_class_grid(initial),
        "final_grid": encode_class_grid(final),
        "per_iteration": {
            "team_reward": [0.1] * 5,
            "urban_fraction": [0.1] * 5,
            "n_conversions": [1] * 5,
            "n_proposals": [2] * 5,
            "n_approved": [1] * 5,
            "n_candidates": [4] * 5,
            "conversion_alignment": [0.1] * 5,
            "well_being": [0.2] * 5,
        },
    }


@pytest.fixture()
def campaign(tmp_path: Path) -> Path:
    directory = tmp_path / "campaign_fit"
    records = []
    for scenario in (1, 2, 3, 4):
        for seed in (0, 1):
            for index, method in enumerate(METHODS):
                record = _record(method, scenario, seed, offset=0.02 * index)
                path = run_json_path(directory, SCALE, scenario, method, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record), encoding="utf-8")
                records.append(record)
    summary = aggregate_records(records, metadata={"tag": "fit"})
    write_campaign_documents(directory, summary, records=records)
    return directory


class TestPaperClasses:
    def test_the_eight_published_classes_in_order(self) -> None:
        assert paper_class_names() == (
            "Arable Land",
            "Grassland",
            "Heather Grassland",
            "Littoral Sediment",
            "Freshwater",
            "Urban",
            "Woodland",
            "Suburban",
        )

    def test_the_colours_are_the_published_ones(self) -> None:
        colours = paper_class_colours()
        assert colours["Arable Land"] == "#d29b38"
        assert colours["Grassland"] == "#9be44d"
        assert colours["Heather Grassland"] == "#7854c7"
        assert colours["Littoral Sediment"] == "#4e8b57"
        assert colours["Freshwater"] == "#3184e0"
        assert colours["Urban"] == "#df6458"
        assert colours["Woodland"] == "#6d6d6d"
        assert colours["Suburban"] == "#a0a0a0"

    def test_suburban_has_no_aggregate_class_behind_it(self) -> None:
        suburban = next(c for c in PAPER_CLASSES if c.name == "Suburban")
        assert suburban.aggregates == ()
        assert suburban.separable is False

    def test_grassland_merges_improved_and_semi_natural(self) -> None:
        mapping = aggregate_to_paper()
        assert mapping[band_index(AggregateClass.IMPROVED_GRASSLAND)] == "Grassland"
        assert mapping[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] == "Grassland"

    def test_saltwater_is_excluded(self) -> None:
        assert band_index(AggregateClass.SALTWATER) not in aggregate_to_paper()

    def test_woodland_merges_broadleaf_and_coniferous(self) -> None:
        mapping = aggregate_to_paper()
        assert mapping[band_index(AggregateClass.BROADLEAF_WOODLAND)] == "Woodland"
        assert mapping[band_index(AggregateClass.CONIFEROUS_WOODLAND)] == "Woodland"


class TestPaperClassGrid:
    def test_excluded_classes_become_minus_one(self) -> None:
        values = np.array([[band_index(AggregateClass.SALTWATER), -1]], dtype=np.int64)
        assert paper_class_grid(values).tolist() == [[-1, -1]]

    def test_a_known_class_maps_to_its_paper_index(self) -> None:
        values = np.array([[band_index(AggregateClass.ARABLE)]], dtype=np.int64)
        assert paper_class_grid(values).tolist() == [[0]]


class TestLandPreference:
    def test_a_uniform_grid_has_one_full_slice(self) -> None:
        arable = band_index(AggregateClass.ARABLE)
        rows = land_preference_table({"initial": np.full((5, 5), arable, np.int64)})
        shares = {
            row["neighbour_class"]: float(row["share"])
            for row in rows
            if row["centre_class"] == "Arable Land"
        }
        assert shares["Arable Land"] == pytest.approx(1.0)
        assert shares["Urban"] == pytest.approx(0.0)

    def test_a_checkerboard_splits_the_neighbourhood(self) -> None:
        arable = band_index(AggregateClass.ARABLE)
        urban = band_index(AggregateClass.BUILT_UP)
        values = np.indices((6, 6)).sum(axis=0) % 2
        grid = np.where(values == 0, arable, urban).astype(np.int64)
        rows = land_preference_table({"initial": grid})
        shares = {
            row["neighbour_class"]: float(row["share"])
            for row in rows
            if row["centre_class"] == "Arable Land"
        }
        # Diagonal neighbours share the parity, the four edge ones do not.
        assert shares["Arable Land"] > 0.0
        assert shares["Urban"] > 0.0
        assert shares["Arable Land"] + shares["Urban"] == pytest.approx(1.0)

    def test_sea_cells_are_neither_centres_nor_neighbours(self) -> None:
        arable = band_index(AggregateClass.ARABLE)
        grid = np.full((4, 4), arable, dtype=np.int64)
        grid[0, :] = -1
        rows = land_preference_table({"initial": grid})
        total = sum(
            int(row["count"]) for row in rows if row["centre_class"] == "Arable Land"
        )
        assert total > 0
        assert all(row["neighbour_class"] != "sea" for row in rows)

    def test_one_figure_per_class_plus_the_legend(self, tmp_path: Path) -> None:
        arable = band_index(AggregateClass.ARABLE)
        results = land_preference_figures(
            {"initial": np.full((6, 6), arable, np.int64)}, tmp_path / "figures"
        )
        names = [result.name for result in results]
        assert len(results) == len(PAPER_CLASSES) + 1
        assert "land_preference_arable_land" in names
        assert LEGEND_NAME in names
        for result in results:
            for path in result.paths:
                assert path.is_file()

    def test_the_table_covers_every_stage(self, tmp_path: Path) -> None:
        arable = band_index(AggregateClass.ARABLE)
        urban = band_index(AggregateClass.BUILT_UP)
        rows = land_preference_table(
            {
                "initial": np.full((4, 4), arable, np.int64),
                "final": np.full((4, 4), urban, np.int64),
            }
        )
        assert {row["stage"] for row in rows} == {"initial", "final"}

    def test_the_suburban_note_is_attached(self, tmp_path: Path) -> None:
        arable = band_index(AggregateClass.ARABLE)
        results = land_preference_figures(
            {"initial": np.full((4, 4), arable, np.int64)}, tmp_path / "figures"
        )
        suburban = next(r for r in results if r.name.endswith("suburban"))
        assert any("aggregate" in note for note in suburban.notes)


class TestRegionMaps:
    def _grids(self) -> tuple[np.ndarray, np.ndarray]:
        arable = band_index(AggregateClass.ARABLE)
        urban = band_index(AggregateClass.BUILT_UP)
        initial = np.full((6, 8), arable, dtype=np.int64)
        initial[0, :] = -1
        final = initial.copy()
        final[2, :] = urban
        return initial, final

    def test_before_and_after_are_written(self, tmp_path: Path) -> None:
        initial, final = self._grids()
        result = region_before_after(initial, final, tmp_path / "f", "SE_region")
        assert result.name == "region_before_after_SE_region"
        assert all(path.is_file() for path in result.paths)

    def test_the_converted_variant_counts_the_conversions(self, tmp_path: Path) -> None:
        initial, final = self._grids()
        result = region_converted(initial, final, tmp_path / "f", "SE_region")
        assert result.name == "region_converted_SE_region"
        assert "8 parcels" in result.caption

    def test_the_paper_orientation_is_the_default(self, tmp_path: Path) -> None:
        initial, final = self._grids()
        result = region_before_after(initial, final, tmp_path / "f", "t")
        assert any("west is at the top" in note for note in result.notes)

    def test_the_raster_orientation_is_available(self, tmp_path: Path) -> None:
        initial, final = self._grids()
        result = region_before_after(
            initial, final, tmp_path / "f", "t", orientation="raster"
        )
        assert any("north up" in note for note in result.notes)

    def test_an_unknown_orientation_is_rejected(self, tmp_path: Path) -> None:
        initial, final = self._grids()
        with pytest.raises(ValueError, match="orientation"):
            region_before_after(
                initial, final, tmp_path / "f", "t", orientation="sideways"
            )


class TestScenarioComparison:
    def test_the_published_layout_is_written(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        loaded = load_campaign(campaign)
        result = scenario_comparison(loaded, tmp_path / "f")
        assert result.name == "scenario_comparison_fit"
        assert all(path.is_file() for path in result.paths)

    def test_the_x_order_is_the_published_one(self) -> None:
        assert PUBLISHED_X_ORDER == ("ltos", "pmaddpg", "mitm", "qolu_buia")

    def test_the_legend_wording_is_the_published_one(self) -> None:
        assert PUBLISHED_METRIC_LEGEND["woodland"] == "Woodland Loss"

    def test_the_csv_carries_three_decimals_and_the_se(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        loaded = load_campaign(campaign)
        result = scenario_comparison(loaded, tmp_path / "f")
        csv_path = next(p for p in result.paths if p.suffix == ".csv")
        text = csv_path.read_text(encoding="utf-8")
        assert "legend" in text.splitlines()[0]
        body = text.splitlines()[1]
        assert len(body.split(",")[5].split(".")[1]) == 3

    def test_the_default_view_is_absolute(self, campaign: Path, tmp_path: Path) -> None:
        loaded = load_campaign(campaign)
        result = scenario_comparison(loaded, tmp_path / "f")
        csv_path = next(p for p in result.paths if p.suffix == ".csv")
        assert ",absolute," in csv_path.read_text(encoding="utf-8")

    def test_an_unknown_view_is_rejected(self, campaign: Path, tmp_path: Path) -> None:
        loaded = load_campaign(campaign)
        with pytest.raises(ValueError, match="view"):
            scenario_comparison(loaded, tmp_path / "f", view="nonsense")

    def test_without_matplotlib_only_the_csv_is_written(
        self, campaign: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(backend, "matplotlib_available", lambda: False)
        loaded = load_campaign(campaign)
        result = scenario_comparison(loaded, tmp_path / "f")
        assert {path.suffix for path in result.paths} == {".csv"}


class TestCli:
    def test_the_region_flag_adds_the_published_figures(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "figures"
        assert main(["--campaign", str(campaign), "--region", "--out", str(out)]) == 0
        names = {path.stem for path in out.iterdir() if path.suffix == ".csv"}
        assert "scenario_comparison_fit" in names
        assert "region_before_after_fit" in names
        assert "region_converted_fit" in names
        assert "land_preference" in names
        index = (out / "figures_index.md").read_text(encoding="utf-8")
        assert "land_preference_arable_land" in index
        assert LEGEND_NAME in index

    def test_without_the_region_flag_the_region_figures_are_absent(
        self, campaign: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "figures"
        assert main(["--campaign", str(campaign), "--out", str(out)]) == 0
        assert not (out / "region_before_after_fit.csv").exists()


class TestVariantColumnOrder:
    """A variant method keeps its place on the published x axis.

    Filtering ``PUBLISHED_X_ORDER`` by literal membership would draw a campaign
    running ``pmaddpg_reg_only`` and ``mitm_learned`` in the column order
    LToS, QOLU+BUIA, P-MADDPG, MitM - which is not the published order, and the
    published order is the whole point of this figure.
    """

    @staticmethod
    def _blocks(methods: tuple[str, ...]) -> dict:
        return {
            1: {
                "scenario": 1,
                "methods": {
                    method: {"absolute": {"land_use": {"mean": 0.5, "se": 0.0}}}
                    for method in methods
                },
            }
        }

    def test_variants_take_their_parents_place(self) -> None:
        from chapter4_qolu.figures.scenario_comparison import _methods

        order = _methods(
            self._blocks(("qolu_buia", "mitm_learned", "ltos", "pmaddpg_reg_only"))
        )
        assert order == ("ltos", "pmaddpg_reg_only", "mitm_learned", "qolu_buia")

    def test_the_published_four_are_unaffected(self) -> None:
        from chapter4_qolu.figures.scenario_comparison import _methods

        order = _methods(self._blocks(("qolu_buia", "mitm", "ltos", "pmaddpg")))
        assert order == PUBLISHED_X_ORDER

    def test_an_unknown_method_goes_last(self) -> None:
        from chapter4_qolu.figures.scenario_comparison import _methods

        order = _methods(self._blocks(("qolu_buia", "ltos", "something_else")))
        assert order[-1] == "something_else"

    def test_the_no_regulator_ablation_comes_first_as_in_the_tables(self) -> None:
        from chapter4_qolu.figures.scenario_comparison import _methods

        order = _methods(
            self._blocks(
                ("qolu_buia", "mitm_learned", "no_regulator", "ltos", "pmaddpg")
            )
        )
        assert order == (
            "no_regulator",
            "ltos",
            "pmaddpg",
            "mitm_learned",
            "qolu_buia",
        )
