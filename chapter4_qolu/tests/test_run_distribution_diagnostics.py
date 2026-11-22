"""End-to-end tests for the Q-distribution diagnostics CLI."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.diagnostics.plan import campaign_agent_directory
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.run_distribution_diagnostics import (
    DiagnosticsSettings,
    _build_parser,
    diagnostics_paths,
    main,
    run_distribution_diagnostics,
)
from chapter4_qolu.tests.raster_fixtures import write_geotiff


@pytest.fixture()
def tiny_class_raster(tmp_path: Path) -> Path:
    rng = np.random.default_rng(20260929)
    codes = rng.integers(1, 11, size=(24, 24), dtype=np.uint8)
    return write_geotiff(
        tmp_path / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        codes,
        origin_e_m=400_000.0,
        origin_n_m=124_000.0,
    )


@pytest.fixture()
def tiny_settings(tiny_class_raster: Path, tmp_path: Path) -> DiagnosticsSettings:
    return DiagnosticsSettings(
        raster=tiny_class_raster,
        tag="tiny",
        window_label="E402_N110_6x6",
        scale=ScaleConfig(grid_size=6, conversion_fraction=1.0, max_conversions=1),
        scenario=1,
        seed=0,
        stakeholders=1,
        iterations=2,
        eval_iterations=2,
        output_dir=tmp_path / "results",
    )


class TestCommandLine:
    def test_help_exits_cleanly(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as error:
            main(["--help"])
        assert error.value.code == 0
        assert "--dry-run" in capsys.readouterr().out

    def test_the_parser_offers_the_house_flags(self) -> None:
        options = {
            action.option_strings[0]
            for action in _build_parser()._actions
            if action.option_strings
        }
        assert {"--dry-run", "--resume", "--out"} <= options


class TestDryRun:
    def test_a_dry_run_writes_nothing(self, tiny_settings: DiagnosticsSettings) -> None:
        plan = run_distribution_diagnostics(tiny_settings, dry_run=True)
        paths = diagnostics_paths(tiny_settings)
        assert not paths.summary.exists()
        assert not paths.report.exists()
        assert plan["dry_run"] is True
        assert plan["plan"]["training_iterations"] == 2


class TestEndToEnd:
    def test_it_writes_a_summary_a_report_and_a_figure(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        summary = run_distribution_diagnostics(tiny_settings)
        paths = diagnostics_paths(tiny_settings)
        assert paths.summary.is_file()
        assert paths.report.is_file()
        assert (paths.figures / "distribution_calibration_tiny.csv").is_file()
        assert summary["tag"] == "tiny"

    def test_the_summary_answers_both_questions(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        summary = run_distribution_diagnostics(tiny_settings)
        assert summary["policy_use_of_the_distribution"]["depends_only_on_the_mean"]
        stakeholder = summary["stakeholders"][0]
        targets = stakeholder["targets"]
        assert set(targets) == {"n_step_reward_sum", "n_step_target", "tail_return"}
        headline = targets[summary["headline_target"]]
        for key in (
            "mae",
            "rmse",
            "pit",
            "coverage",
            "crps",
            "boundary_mass",
            "support",
        ):
            assert key in headline
        assert "resolves_the_signal" in headline["support"]
        assert "climatology" in headline["crps"]["skill"]
        assert "initialisation" in headline["crps"]["skill"]

    def test_the_verdicts_are_stated_either_way(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        summary = run_distribution_diagnostics(tiny_settings)
        verdict = summary["verdict"]
        assert set(verdict) >= {
            "state_conditional",
            "calibrated",
            "support_adequate",
            "support_resolves_the_signal",
            "statement",
        }
        assert isinstance(verdict["statement"], str) and verdict["statement"]

    def test_the_report_names_the_headline_number(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        run_distribution_diagnostics(tiny_settings)
        report = diagnostics_paths(tiny_settings).report.read_text(encoding="utf-8")
        assert "CRPS skill" in report
        assert "boundary" in report.lower()
        assert "E[Z]" in report
        assert "atom spacing" in report
        assert "mean interval width" in report
        assert "regressed onto" in report

    def test_the_figure_csv_carries_both_panels(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        run_distribution_diagnostics(tiny_settings)
        paths = diagnostics_paths(tiny_settings)
        rows = (
            (paths.figures / "distribution_calibration_tiny.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        panels = {line.split(",")[0] for line in rows[1:]}
        assert panels == {"pit_histogram", "reliability"}

    def test_the_final_agents_are_persisted(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        run_distribution_diagnostics(tiny_settings)
        paths = diagnostics_paths(tiny_settings)
        assert (paths.agents / "stakeholder0.pt").is_file()


class TestAgentSources:
    def test_a_campaign_resolves_to_the_runs_own_agent_folder(
        self, tiny_settings: DiagnosticsSettings, tmp_path: Path
    ) -> None:
        settings = replace(tiny_settings, campaign=tmp_path / "campaign_main")
        directory = campaign_agent_directory(settings)
        assert directory is not None
        assert directory.parts[-3:] == (
            "6x6_f1.0_cap1",
            "scenario1",
            "qolu_buia_seed0.agents",
        )

    def test_a_campaign_without_persisted_agents_trains_instead(
        self, tiny_settings: DiagnosticsSettings, tmp_path: Path
    ) -> None:
        settings = replace(tiny_settings, campaign=tmp_path / "campaign_main")
        summary = run_distribution_diagnostics(settings)
        assert summary["agents"]["source"] == "trained"

    def test_an_explicit_checkpoint_directory_is_diagnosed_as_is(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        run_distribution_diagnostics(tiny_settings)
        agents = diagnostics_paths(tiny_settings).agents
        settings = replace(tiny_settings, tag="reused", checkpoint=agents)
        summary = run_distribution_diagnostics(settings)
        assert summary["agents"]["source"] == "checkpoint"
        assert summary["agents"]["directory"] == str(agents)
        assert summary["agents"]["n_checkpoints"] == 1

    def test_the_investor_of_the_rollout_is_always_a_fresh_one(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        """BUIA is not checkpointed, so the rollout's proposer is never trained.

        ``run_distribution_diagnostics`` builds the evaluation investor with
        ``investor_for(env, spec)`` whichever way the stakeholders arrived, so a
        run that has just trained one must not claim otherwise.
        """
        trained = run_distribution_diagnostics(tiny_settings)
        assert trained["agents"]["source"] == "trained"
        assert trained["agents"]["investor"] == "fresh"
        agents = diagnostics_paths(tiny_settings).agents
        reused = run_distribution_diagnostics(
            replace(tiny_settings, tag="reused", checkpoint=agents)
        )
        assert reused["agents"]["investor"] == "fresh"


class TestResume:
    def test_resume_reuses_a_finished_summary(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        first = run_distribution_diagnostics(tiny_settings)
        again = run_distribution_diagnostics(replace(tiny_settings, resume=True))
        assert again["generated"] == first["generated"]

    def test_resume_re_renders_the_report_from_the_stored_summary(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        """The JSON is the record; the markdown is a view of it.

        Re-rendering on resume costs nothing - no rollout, no training - and
        means a document can be brought up to date with the reporting code
        without re-running an experiment or hand-editing a result file.
        """
        run_distribution_diagnostics(tiny_settings)
        paths = diagnostics_paths(tiny_settings)
        paths.report.write_text("stale", encoding="utf-8")
        summary = run_distribution_diagnostics(replace(tiny_settings, resume=True))
        report = paths.report.read_text(encoding="utf-8")
        assert report != "stale"
        assert summary["verdict"]["statement"] in report

    def test_without_resume_the_summary_is_rebuilt(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        run_distribution_diagnostics(tiny_settings)
        paths = diagnostics_paths(tiny_settings)
        before = json.loads(paths.summary.read_text(encoding="utf-8"))
        again = run_distribution_diagnostics(tiny_settings)
        assert again["headline_target"] == before["headline_target"]

    def test_resume_reuses_the_persisted_agents_instead_of_retraining(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        run_distribution_diagnostics(tiny_settings)
        paths = diagnostics_paths(tiny_settings)
        stamp = (paths.agents / "stakeholder0.pt").stat().st_mtime_ns
        paths.summary.unlink()
        summary = run_distribution_diagnostics(replace(tiny_settings, resume=True))
        assert (paths.agents / "stakeholder0.pt").stat().st_mtime_ns == stamp
        assert summary["agents"]["source"] == "checkpoint"
