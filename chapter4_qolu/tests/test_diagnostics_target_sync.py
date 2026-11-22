"""Making ``target_sync_interval`` settable per run, by the *existing* route.

``t_n`` is a free choice of the implementation, and a value longer than a
campaign run's whole gradient budget means the mechanism never fires.  Deciding
what ``t_n`` should be needs an experiment, and an experiment needs the value
to be settable without editing ``config.py``.

There is already one route by which a :class:`~chapter4_qolu.config.QOLUConfig`
field is moved off its default for a single run:
:mod:`chapter4_qolu.hparam_worker` perturbs it after
:func:`chapter4_qolu.scenario_worker.qolu_config_for` has resolved the method
variant.  These tests pin that this stays *one* mechanism - the override is
applied inside ``qolu_config_for`` and both callers go through it - and that
the diagnostics driver, which is what measures the evaluation-time approval
rate, can ask for it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.diagnostics.arguments import build_parser, settings_from
from chapter4_qolu.diagnostics.sweep import cell_settings
from chapter4_qolu.diagnostics.training import learner_record, training_record
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.run_distribution_diagnostics import (
    DiagnosticsSettings,
    diagnostics_paths,
    run_distribution_diagnostics,
)
from chapter4_qolu.scenario_worker import qolu_config_for
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
        iterations=30,
        eval_iterations=2,
        qolu_reward_mode="team",
        output_dir=tmp_path / "results",
    )


class TestTheOverrideGoesThroughTheExistingResolver:
    def test_without_an_override_the_config_is_the_package_default(self) -> None:
        config = qolu_config_for("qolu_buia", observation_channels=3, window=9)
        assert config.target_sync_interval == 25

    def test_an_override_moves_exactly_the_field_it_names(self) -> None:
        config = qolu_config_for(
            "qolu_buia",
            observation_channels=3,
            window=9,
            overrides={"target_sync_interval": 1_000},
        )
        default = qolu_config_for("qolu_buia", observation_channels=3, window=9)
        assert config.target_sync_interval == 1_000
        assert replace(config, target_sync_interval=25) == default

    def test_an_empty_override_is_the_default_config(self) -> None:
        assert qolu_config_for(
            "qolu_buia", observation_channels=3, window=9, overrides={}
        ) == qolu_config_for("qolu_buia", observation_channels=3, window=9)

    def test_the_method_variant_still_decides_the_switches(self) -> None:
        config = qolu_config_for(
            "qolu_c51",
            observation_channels=3,
            window=9,
            overrides={"target_sync_interval": 1_000},
        )
        assert config.noisy_nets is False
        assert config.target_sync_interval == 1_000

    def test_an_unknown_field_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="not_a_field"):
            qolu_config_for(
                "qolu_buia",
                observation_channels=3,
                window=9,
                overrides={"not_a_field": 1},
            )

    def test_an_invalid_value_is_still_validated(self) -> None:
        with pytest.raises(ValueError, match="target_sync_interval must be positive"):
            qolu_config_for(
                "qolu_buia",
                observation_channels=3,
                window=9,
                overrides={"target_sync_interval": 0},
            )


class TestTheDiagnosticsSettingsCarryIt:
    def test_the_default_is_none_meaning_the_package_default(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        assert tiny_settings.target_sync_interval is None
        assert tiny_settings.qolu_overrides == {}

    def test_a_value_becomes_a_qolu_override(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        settings = replace(tiny_settings, target_sync_interval=25)
        assert settings.qolu_overrides == {"target_sync_interval": 25}

    def test_a_non_positive_interval_is_refused(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        with pytest.raises(ValueError, match="target_sync_interval must be positive"):
            replace(tiny_settings, target_sync_interval=0)

    def test_the_record_states_what_was_asked_for(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        assert tiny_settings.as_record()["target_sync_interval"] is None
        assert (
            replace(tiny_settings, target_sync_interval=25).as_record()[
                "target_sync_interval"
            ]
            == 25
        )

    def test_a_sweep_cell_inherits_it(self, tiny_settings: DiagnosticsSettings) -> None:
        base = replace(tiny_settings, target_sync_interval=25)
        assert cell_settings(base, scenario=2, seed=1).target_sync_interval == 25


class TestTheCommandLineOffersIt:
    def test_the_flag_exists(self) -> None:
        options = {
            action.option_strings[0]
            for action in build_parser()._actions
            if action.option_strings
        }
        assert "--target-sync-interval" in options

    def test_it_defaults_to_the_package_default(self, tiny_class_raster: Path) -> None:
        arguments = build_parser().parse_args(["--raster", str(tiny_class_raster)])
        assert settings_from(arguments).target_sync_interval is None

    def test_it_reaches_the_settings(self, tiny_class_raster: Path) -> None:
        arguments = build_parser().parse_args(
            ["--raster", str(tiny_class_raster), "--target-sync-interval", "25"]
        )
        assert settings_from(arguments).target_sync_interval == 25


class TestTheRecordsSayWhatHappened:
    def test_a_learner_record_reports_the_interval_and_the_syncs(self) -> None:
        from chapter4_qolu.agent.qolu_agent import QOLUAgent

        config = qolu_config_for(
            "qolu_buia",
            observation_channels=3,
            window=3,
            overrides={"target_sync_interval": 4},
        )
        agents = [QOLUAgent(config, seed=0)]
        agents[0].sync_target()
        record = learner_record(agents)
        assert record["target_sync_interval"] == 4
        assert record["target_syncs"] == [1]
        assert record["total_target_syncs"] == 1
        assert record["target_network_ever_synced"] is True

    def test_a_learner_record_flags_a_target_that_never_synced(self) -> None:
        from chapter4_qolu.agent.qolu_agent import QOLUAgent

        config = qolu_config_for("qolu_buia", observation_channels=3, window=3)
        record = learner_record([QOLUAgent(config, seed=0)])
        assert record["total_target_syncs"] == 0
        assert record["target_network_ever_synced"] is False


class TestEndToEnd:
    def test_the_summary_records_the_interval_the_run_used(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        settings = replace(tiny_settings, target_sync_interval=5)
        summary = run_distribution_diagnostics(settings)
        assert summary["learner"]["target_sync_interval"] == 5
        assert summary["run"]["target_sync_interval"] == 5

    def test_a_short_interval_actually_fires_the_reset(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        summary = run_distribution_diagnostics(
            replace(tiny_settings, target_sync_interval=2)
        )
        assert summary["learner"]["total_target_syncs"] > 0
        assert summary["learner"]["target_network_ever_synced"] is True

    def test_an_interval_longer_than_the_run_fires_nothing(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        summary = run_distribution_diagnostics(
            replace(tiny_settings, target_sync_interval=1_000)
        )
        assert summary["learner"]["target_sync_interval"] == 1_000
        assert summary["learner"]["total_target_syncs"] == 0
        assert summary["learner"]["target_network_ever_synced"] is False

    def test_a_default_run_fires_the_reset(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        """At the package default ``theta^- <- theta`` fires inside an ordinary run.

        ``learn()`` waits for ``learning_starts`` = 1000 stored transitions,
        i.e. 20 iterations of 50 appraisals, so 60 iterations give about 40
        gradient steps against the default interval of 25.  No override is
        set: the run takes the package default through the usual route.
        """
        settings = replace(tiny_settings, iterations=60)
        summary = run_distribution_diagnostics(settings)
        learner = summary["learner"]
        assert summary["run"]["target_sync_interval"] is None
        assert learner["target_sync_interval"] == 25
        assert learner["total_target_syncs"] >= 1
        assert learner["target_network_ever_synced"] is True
        report = diagnostics_paths(settings).report.read_text(encoding="utf-8")
        assert "`t_n` = 25" in report
        assert "**never** synchronised" not in report

    def test_the_summary_carries_the_training_outcome(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        """The metrics the chapter reports, beside the diagnostics of the same run.

        Without this the approval rate and the published metrics would come
        from two different runs of the same settings, and only one of them
        would be on disk.
        """
        summary = run_distribution_diagnostics(
            replace(tiny_settings, target_sync_interval=5)
        )
        training = summary["training"]
        assert training["source"] == "trained"
        assert training["iterations"] == 30
        assert isinstance(training["total_conversions"], int)
        assert set(training["final_metrics"]) >= {"land_use", "woodland", "urban"}
        assert set(training["final_metrics_absolute"]) >= {
            "land_use",
            "woodland",
            "urban",
        }

    def test_agents_loaded_from_disk_report_no_training(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        run_distribution_diagnostics(tiny_settings)
        agents = diagnostics_paths(tiny_settings).agents
        summary = run_distribution_diagnostics(
            replace(tiny_settings, tag="reused", checkpoint=agents)
        )
        assert summary["training"]["source"] == "checkpoint"
        assert "total_conversions" not in summary["training"]
        assert summary["learner"]["target_sync_interval"] == 25

    def test_the_approval_rate_is_reported_per_stakeholder(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        """The decisive number of the experiment has to be readable off the JSON."""
        summary = run_distribution_diagnostics(tiny_settings)
        path = diagnostics_paths(tiny_settings).summary
        stored = json.loads(path.read_text(encoding="utf-8"))
        rates = [entry["approval_rate"] for entry in stored["rollout"]["stakeholders"]]
        assert len(rates) == 1
        assert all(0.0 <= rate <= 1.0 for rate in rates)
        assert summary["rollout"]["stakeholders"][0]["approval_rate"] == rates[0]

    def test_the_report_states_when_the_target_never_synced(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        settings = replace(tiny_settings, target_sync_interval=1_000)
        run_distribution_diagnostics(settings)
        report = diagnostics_paths(settings).report.read_text(encoding="utf-8")
        assert "`t_n` = 1000" in report
        assert "**never** synchronised their target networks" in report
        assert "randomly initialised weights" in report

    def test_the_report_counts_syncs_per_stakeholder_not_in_total(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        """980 gradient steps is a *per agent* budget, so the sync count must be too.

        A line reading "synchronised 117 times over 980 gradient steps" silently
        divides a three-stakeholder total by a one-stakeholder denominator.
        """
        summary = run_distribution_diagnostics(
            replace(tiny_settings, stakeholders=3, target_sync_interval=5)
        )
        report = diagnostics_paths(
            replace(tiny_settings, stakeholders=3, target_sync_interval=5)
        ).report.read_text(encoding="utf-8")
        per_agent = summary["learner"]["target_syncs"][0]
        steps = summary["learner"]["gradient_steps"][0]
        assert summary["learner"]["total_target_syncs"] == 3 * per_agent
        assert f"{per_agent} time(s) each" in report
        assert f"{steps} gradient steps each" in report
        assert (
            str(summary["learner"]["total_target_syncs"])
            not in report.split("Target-network schedule")[1].split("\n")[0]
        )


class TestTrainingRecordShape:
    def test_it_refuses_to_invent_a_report(self) -> None:
        assert training_record(None, None, None) == {"source": "checkpoint"}
