"""Tests for the four-scenario campaign runner and its result paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chapter4_qolu.config import EnvConfig
from chapter4_qolu.reporting.paths import (
    ScaleConfig,
    campaign_directory,
    is_complete,
    parse_scale_config,
    run_json_path,
    should_run,
)
from chapter4_qolu.run_scenarios import (
    DEFAULT_METHODS,
    METHOD_COST_SECONDS,
    METHOD_KEYS,
    CampaignSettings,
    _build_parser,
    _settings_from,
    campaign_runs,
    estimate_runtime_seconds,
    format_duration,
    pending_runs,
    run_campaign,
    smoke_settings,
)
from chapter4_qolu.scenario_worker import RunSpec, buia_proposal_limit, execute_run


class TestUncappedConfiguration:
    """``cap none`` has to work for QOLU + BUIA, not only for the baselines."""

    def test_proposal_limit_is_the_cap_when_there_is_one(self) -> None:
        assert buia_proposal_limit(EnvConfig(max_conversions_per_iteration=10)) == 10

    def test_proposal_limit_falls_back_to_the_decision_buffer(self) -> None:
        config = EnvConfig(max_conversions_per_iteration=None, samples_per_iteration=50)
        assert buia_proposal_limit(config) == 50

    def test_runs_qolu_buia_without_a_conversion_cap(
        self, lcm_tif_path: Path, tmp_path: Path
    ) -> None:
        spec = RunSpec(
            raster=lcm_tif_path,
            method="qolu_buia",
            scenario=1,
            seed=0,
            scale=parse_scale_config("12:1.0:none"),
            output_path=tmp_path / "qolu_buia_seed0.json",
            iterations=3,
            stakeholders=1,
        )
        record = execute_run(spec)
        assert record["env_settings"]["max_conversions_per_iteration"] is None
        assert spec.output_path.is_file()
        assert len(record["per_iteration"]["n_conversions"]) == 3


class TestScaleConfig:
    def test_parses_a_grid_fraction_cap_triple(self) -> None:
        scale = parse_scale_config("50:1.0:10")
        assert scale == ScaleConfig(
            grid_size=50, conversion_fraction=1.0, max_conversions=10
        )

    def test_none_means_no_conversion_cap(self) -> None:
        assert parse_scale_config("100:0.25:none").max_conversions is None
        assert parse_scale_config("100:0.25:NONE").max_conversions is None

    def test_label_is_stable_and_readable(self) -> None:
        assert parse_scale_config("50:1.0:10").label == "50x50_f1.0_cap10"
        assert parse_scale_config("100:0.25:10").label == "100x100_f0.25_cap10"
        assert parse_scale_config("30:0.5:none").label == "30x30_f0.5_capnone"

    def test_rejects_malformed_triples(self) -> None:
        for text in ("50:1.0", "50:1.0:10:4", "x:1.0:10", "50:two:10", "50:1.0:cap"):
            with pytest.raises(ValueError):
                parse_scale_config(text)

    def test_rejects_out_of_range_values(self) -> None:
        for text in ("0:1.0:10", "50:0:10", "50:1.5:10", "50:1.0:-1"):
            with pytest.raises(ValueError):
                parse_scale_config(text)

    def test_as_record_round_trips_into_json(self) -> None:
        scale = parse_scale_config("50:1.0:none")
        record = scale.as_record()
        assert json.loads(json.dumps(record)) == record
        assert record["max_conversions_per_iteration"] is None


class TestResultPaths:
    def test_follows_the_documented_layout(self, tmp_path: Path) -> None:
        directory = campaign_directory(tmp_path, "demo")
        assert directory == tmp_path / "campaign_demo"
        path = run_json_path(
            directory, parse_scale_config("50:1.0:10"), 3, "qolu_buia", 7
        )
        assert path == (
            tmp_path
            / "campaign_demo"
            / "50x50_f1.0_cap10"
            / "scenario3"
            / "qolu_buia_seed7.json"
        )

    def test_a_missing_file_is_not_complete(self, tmp_path: Path) -> None:
        assert is_complete(tmp_path / "absent.json") is False

    def test_a_truncated_file_is_not_complete(self, tmp_path: Path) -> None:
        path = tmp_path / "half.json"
        path.write_text('{"method": "mitm", "final_met', encoding="utf-8")
        assert is_complete(path) is False

    def test_a_record_without_final_metrics_is_not_complete(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({"method": "mitm"}), encoding="utf-8")
        assert is_complete(path) is False

    def test_a_finished_record_is_complete(self, tmp_path: Path) -> None:
        path = tmp_path / "done.json"
        path.write_text(
            json.dumps({"method": "mitm", "final_metrics": {"land_use": 0.1}}),
            encoding="utf-8",
        )
        assert is_complete(path) is True

    def test_resume_skips_only_finished_records(self, tmp_path: Path) -> None:
        path = tmp_path / "done.json"
        path.write_text(
            json.dumps({"method": "mitm", "final_metrics": {}}), encoding="utf-8"
        )
        assert should_run(path, resume=True) is False
        assert should_run(path, resume=False) is True
        assert should_run(tmp_path / "absent.json", resume=True) is True


class TestCampaignExpansion:
    @pytest.fixture()
    def settings(self, tmp_path: Path) -> CampaignSettings:
        return CampaignSettings(
            raster=tmp_path / "raster.tif",
            tag="demo",
            methods=("qolu_buia", "mitm"),
            scenarios=(1, 2),
            seeds=(0, 1, 2),
            configs=(
                parse_scale_config("50:1.0:10"),
                parse_scale_config("30:0.5:none"),
            ),
            output_dir=tmp_path,
        )

    def test_covers_the_full_cross_product(self, settings: CampaignSettings) -> None:
        specs = campaign_runs(settings)
        assert len(specs) == 2 * 2 * 3 * 2
        assert len({spec.output_path for spec in specs}) == len(specs)

    def test_every_method_in_a_cell_shares_scale_and_seed(
        self, settings: CampaignSettings
    ) -> None:
        cell = [
            spec
            for spec in campaign_runs(settings)
            if spec.scenario == 2
            and spec.seed == 1
            and spec.config_label.startswith("50")
        ]
        assert {spec.method for spec in cell} == {"qolu_buia", "mitm"}
        shared = {
            (
                spec.grid_size,
                spec.conversion_fraction,
                spec.max_conversions,
                spec.seed,
                spec.iterations,
            )
            for spec in cell
        }
        assert len(shared) == 1

    def test_default_methods_are_the_published_four(self) -> None:
        assert DEFAULT_METHODS == ("qolu_buia", "pmaddpg", "ltos", "mitm")
        assert set(DEFAULT_METHODS) <= set(METHOD_KEYS)

    def test_rejects_an_unknown_method(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            CampaignSettings(raster=tmp_path / "r.tif", methods=("nope",))

    def test_rejects_an_unknown_scenario(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            CampaignSettings(raster=tmp_path / "r.tif", scenarios=(9,))

    def test_rejects_an_empty_seed_list(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            CampaignSettings(raster=tmp_path / "r.tif", seeds=())

    def test_pending_runs_partitions_on_existing_records(
        self, settings: CampaignSettings
    ) -> None:
        specs = campaign_runs(settings)
        done = specs[0]
        done.output_path.parent.mkdir(parents=True, exist_ok=True)
        done.output_path.write_text(json.dumps({"final_metrics": {}}), encoding="utf-8")
        todo, skipped = pending_runs(specs, resume=True)
        assert skipped == (done,)
        assert len(todo) == len(specs) - 1
        todo, skipped = pending_runs(specs, resume=False)
        assert skipped == ()
        assert len(todo) == len(specs)


class TestRuntimeEstimate:
    def test_uses_the_measured_per_method_costs(self, tmp_path: Path) -> None:
        settings = CampaignSettings(
            raster=tmp_path / "r.tif",
            scenarios=(1,),
            seeds=(0,),
            configs=(parse_scale_config("50:1.0:10"),),
            output_dir=tmp_path,
        )
        expected = sum(METHOD_COST_SECONDS[method] for method in DEFAULT_METHODS)
        assert estimate_runtime_seconds(campaign_runs(settings)) == pytest.approx(
            expected
        )

    def test_scales_linearly_with_iterations(self, tmp_path: Path) -> None:
        def total(iterations: int) -> float:
            settings = CampaignSettings(
                raster=tmp_path / "r.tif",
                methods=("mitm",),
                scenarios=(1,),
                seeds=(0,),
                configs=(parse_scale_config("50:1.0:10"),),
                iterations=iterations,
                output_dir=tmp_path,
            )
            return estimate_runtime_seconds(campaign_runs(settings))

        assert total(500) == pytest.approx(total(1000) / 2.0)

    def test_a_hundred_square_grid_costs_a_quarter_more(self, tmp_path: Path) -> None:
        def total(config: str) -> float:
            settings = CampaignSettings(
                raster=tmp_path / "r.tif",
                methods=("mitm",),
                scenarios=(1,),
                seeds=(0,),
                configs=(parse_scale_config(config),),
                output_dir=tmp_path,
            )
            return estimate_runtime_seconds(campaign_runs(settings))

        assert total("100:0.25:10") == pytest.approx(total("50:1.0:10") * 1.25)

    def test_formats_a_human_readable_duration(self) -> None:
        assert format_duration(45.0) == "45 s"
        assert format_duration(600.0) == "10.0 min"
        assert format_duration(7200.0) == "2.0 h"


class TestSmokeSettings:
    def test_is_the_documented_three_minute_configuration(self, tmp_path: Path) -> None:
        settings = smoke_settings(raster=tmp_path / "r.tif", output_dir=tmp_path)
        assert settings.iterations == 50
        assert settings.seeds == (0,)
        assert settings.scenarios == (1,)
        assert settings.methods == DEFAULT_METHODS
        assert [scale.label for scale in settings.configs] == ["30x30_f1.0_cap10"]
        assert settings.tag == "smoke"


class TestSmokeCampaign:
    """End-to-end smoke run on the placeholder Northern-Ireland raster."""

    @pytest.fixture(scope="class")
    def campaign(
        self, lcm_tif_path: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> tuple[dict[str, object], Path]:
        output_dir = tmp_path_factory.mktemp("campaign")
        settings = smoke_settings(raster=lcm_tif_path, output_dir=output_dir)
        with pytest.warns(UserWarning, match="placeholder"):
            summary = run_campaign(settings)
        return summary, settings.campaign_dir

    def test_writes_one_json_per_run(
        self, campaign: tuple[dict[str, object], Path]
    ) -> None:
        _, directory = campaign
        runs = sorted(directory.glob("*/scenario*/*.json"))
        assert len(runs) == len(DEFAULT_METHODS)
        assert {path.stem for path in runs} == {
            f"{method}_seed0" for method in DEFAULT_METHODS
        }

    def test_each_record_has_the_required_fields(
        self, campaign: tuple[dict[str, object], Path]
    ) -> None:
        _, directory = campaign
        path = directory / "30x30_f1.0_cap10" / "scenario1" / "mitm_seed0.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["method"] == "mitm"
        assert record["scenario"] == 1
        assert record["seed"] == 0
        assert record["iterations"] == 50
        assert record["qolu_reward_mode"] == "per_parcel"
        assert record["runtime_seconds"] >= 0.0
        assert set(record["weights"]) == {"land_use", "woodland", "urban"}
        assert record["env_settings"]["grid_size"] == 30
        assert record["env_settings"]["conversion_fraction"] == 1.0
        assert record["env_settings"]["max_conversions_per_iteration"] == 10
        assert isinstance(record["total_conversions"], int)
        assert record["saturation_iteration"] is None or isinstance(
            record["saturation_iteration"], int
        )
        for series in (
            "team_reward",
            "urban_fraction",
            "n_conversions",
            "n_proposals",
            "n_approved",
            "n_candidates",
            "conversion_alignment",
            "well_being",
        ):
            assert len(record["per_iteration"][series]) == 50
        for view in ("final_metrics", "final_metrics_absolute"):
            assert set(record[view]) == {
                "land_use",
                "woodland",
                "urban",
                "conversion_alignment",
                "c_align",
                "well_being",
            }

    def test_every_method_saw_the_same_landscape_and_settings(
        self, campaign: tuple[dict[str, object], Path]
    ) -> None:
        _, directory = campaign
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*/scenario*/*.json"))
        ]
        assert len({record["initial_grid_digest"] for record in records}) == 1
        assert len({json.dumps(record["env_settings"]) for record in records}) == 1
        assert {record["seed"] for record in records} == {0}

    def test_writes_the_summary_and_the_three_documents(
        self, campaign: tuple[dict[str, object], Path]
    ) -> None:
        _, directory = campaign
        for name in ("summary.json", "tables.md", "tables.tex", "pattern_check.md"):
            assert (directory / name).is_file()
            assert (directory / name).read_text(encoding="utf-8").strip()
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        assert summary["campaign"]["tag"] == "smoke"
        cell = summary["configurations"][0]["scenarios"][0]
        assert set(cell["methods"]) == set(DEFAULT_METHODS)
        assert set(cell["ranking"]) >= {"land_use", "woodland", "urban"}

    def test_the_returned_summary_matches_the_written_one(
        self, campaign: tuple[dict[str, object], Path]
    ) -> None:
        summary, directory = campaign
        written = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        assert summary == written

    def test_a_single_seed_reports_that_no_test_is_possible(
        self, campaign: tuple[dict[str, object], Path]
    ) -> None:
        summary, _ = campaign
        cell = summary["configurations"][0]["scenarios"][0]
        entry = cell["tests"]["mitm"]["land_use"]
        assert entry["possible"] is False
        assert "no paired t-test" in entry["note"]

    def test_resuming_the_same_campaign_reruns_nothing(
        self, campaign: tuple[dict[str, object], Path], lcm_tif_path: Path
    ) -> None:
        _, directory = campaign
        settings = smoke_settings(
            raster=lcm_tif_path, output_dir=directory.parent
        ).with_resume()
        todo, skipped = pending_runs(campaign_runs(settings), resume=True)
        assert todo == ()
        assert len(skipped) == len(DEFAULT_METHODS)


class TestScenarioWeightSwitches:
    """``--scale-rewards-by-weight`` and ``--weighted-consensus``.

    Both are off by default; a campaign that turns them on is kept apart by its
    ``--tag``.
    """

    def _settings(self, tmp_path: Path, *flags: str) -> CampaignSettings:
        arguments = _build_parser().parse_args(
            ["--tag", "weighted", "--output-dir", str(tmp_path), *flags]
        )
        return _settings_from(arguments, tmp_path / "raster.tif")

    def test_both_switches_default_to_off(self, tmp_path: Path) -> None:
        settings = CampaignSettings(raster=tmp_path / "r.tif", output_dir=tmp_path)
        assert settings.scale_by_scenario_weight is False
        assert settings.weighted_consensus is False

    def test_the_metadata_records_both_switches(self, tmp_path: Path) -> None:
        metadata = CampaignSettings(
            raster=tmp_path / "r.tif", output_dir=tmp_path
        ).metadata()
        assert metadata["scale_by_scenario_weight"] is False
        assert metadata["weighted_consensus"] is False
        assert json.loads(json.dumps(metadata)) == metadata

    def test_the_cli_turns_both_switches_on(self, tmp_path: Path) -> None:
        settings = self._settings(
            tmp_path, "--scale-rewards-by-weight", "--weighted-consensus"
        )
        assert settings.scale_by_scenario_weight is True
        assert settings.weighted_consensus is True
        assert settings.metadata()["scale_by_scenario_weight"] is True
        assert settings.metadata()["weighted_consensus"] is True

    def test_the_two_switches_are_independent(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path, "--weighted-consensus")
        assert settings.scale_by_scenario_weight is False
        assert settings.weighted_consensus is True

    def test_every_run_spec_carries_the_switches(self, tmp_path: Path) -> None:
        settings = self._settings(
            tmp_path, "--scale-rewards-by-weight", "--weighted-consensus"
        )
        specs = campaign_runs(settings)
        assert specs
        assert all(spec.scale_by_scenario_weight for spec in specs)
        assert all(spec.weighted_consensus for spec in specs)

    def test_the_tag_keeps_a_weighted_campaign_in_its_own_folder(
        self, tmp_path: Path
    ) -> None:
        weighted = self._settings(tmp_path, "--weighted-consensus")
        plain = CampaignSettings(
            raster=tmp_path / "raster.tif", tag="main", output_dir=tmp_path
        )
        assert weighted.campaign_dir == tmp_path / "campaign_weighted"
        assert weighted.campaign_dir != plain.campaign_dir
        assert [
            spec.output_path.relative_to(weighted.campaign_dir)
            for spec in campaign_runs(weighted)
        ] == [
            spec.output_path.relative_to(plain.campaign_dir)
            for spec in campaign_runs(plain)
        ]

    def test_the_run_record_carries_the_switches(
        self, lcm_tif_path: Path, tmp_path: Path
    ) -> None:
        spec = RunSpec(
            raster=lcm_tif_path,
            method="qolu_buia",
            scenario=3,
            seed=0,
            scale=parse_scale_config("12:1.0:4"),
            output_path=tmp_path / "qolu_buia_seed0.json",
            iterations=2,
            stakeholders=2,
            scale_by_scenario_weight=True,
            weighted_consensus=True,
        )
        record = execute_run(spec)
        assert record["scale_by_scenario_weight"] is True
        assert record["weighted_consensus"] is True
        saved = json.loads(spec.output_path.read_text(encoding="utf-8"))
        assert saved["scale_by_scenario_weight"] is True
        assert saved["weighted_consensus"] is True

    def test_a_default_run_record_says_the_switches_were_off(
        self, lcm_tif_path: Path, tmp_path: Path
    ) -> None:
        spec = RunSpec(
            raster=lcm_tif_path,
            method="mitm",
            scenario=1,
            seed=0,
            scale=parse_scale_config("12:1.0:4"),
            output_path=tmp_path / "mitm_seed0.json",
            iterations=2,
            stakeholders=2,
        )
        record = execute_run(spec)
        assert record["scale_by_scenario_weight"] is False
        assert record["weighted_consensus"] is False
