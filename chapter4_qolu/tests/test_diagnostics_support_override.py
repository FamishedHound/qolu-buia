"""The atom support and the gradient budget, settable per run by the *existing* route.

Two settings of the implementation are pinned here, for two different reasons.

**The atom support** - ``n_atoms`` / ``v_min`` / ``v_max`` - is *published*
(Table 4.1: 51 atoms over ``[-200, 200]``), so the package default must not
move.  But the achievable return in this environment is tiny beside it: the
``per_parcel`` reward lies in ``[-0.5, 0.5]``, so at ``gamma = 0.99`` and
``n_step = 3`` the whole n-step target range is 2.97 against an atom spacing of
``delta_z = 8``.  Whether the categorical head can resolve a per-parcel
difference at all is therefore an experiment, and an experiment needs the
support to be settable without editing ``config.py``.

**The gradient budget** - how many ``learn()`` calls one environment iteration
fires - is not published at all.  By default the loops in
:mod:`chapter4_qolu.runner` call ``learn()`` once per iteration, which at the
campaign's 1 000 iterations and a 50 000-transition buffer samples each
transition about five times in a whole run.  Raising it is the other half of
the same question, so it is a config field defaulting to 1.

Both go through the one override route that already exists -
:func:`chapter4_qolu.scenario_worker.qolu_config_for` with ``overrides=...``,
which :mod:`chapter4_qolu.hparam_worker` and ``--target-sync-interval`` already
use - so a support-probe run and a sensitivity run of the same settings are the
same experiment.  See ``tests/test_diagnostics_target_sync.py``, which pins the
same property for ``t_n``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.agent.qolu_agent import QOLUAgent
from chapter4_qolu.config import QOLUConfig
from chapter4_qolu.diagnostics.arguments import build_parser, settings_from
from chapter4_qolu.diagnostics.sweep import cell_settings
from chapter4_qolu.diagnostics.training import learner_record
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.run_distribution_diagnostics import (
    DiagnosticsSettings,
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
        qolu_reward_mode="per_parcel",
        output_dir=tmp_path / "results",
    )


class TestThePublishedSupportIsStillTheDefault:
    def test_table_4_1_is_what_config_says(self) -> None:
        config = QOLUConfig(observation_channels=3, observation_window=9)
        assert (config.n_atoms, config.v_min, config.v_max) == (51, -200.0, 200.0)

    def test_the_resolver_hands_back_the_published_support(self) -> None:
        config = qolu_config_for("qolu_buia", observation_channels=3, window=9)
        assert (config.n_atoms, config.v_min, config.v_max) == (51, -200.0, 200.0)


class TestTheSupportOverrideGoesThroughTheExistingResolver:
    def test_it_moves_exactly_the_fields_it_names(self) -> None:
        config = qolu_config_for(
            "qolu_buia",
            observation_channels=3,
            window=9,
            overrides={"v_min": -3.0, "v_max": 3.0, "n_atoms": 51},
        )
        default = qolu_config_for("qolu_buia", observation_channels=3, window=9)
        assert (config.n_atoms, config.v_min, config.v_max) == (51, -3.0, 3.0)
        assert replace(config, v_min=-200.0, v_max=200.0) == default

    def test_an_inverted_support_is_still_validated(self) -> None:
        with pytest.raises(ValueError, match="v_max must be greater than v_min"):
            qolu_config_for(
                "qolu_buia",
                observation_channels=3,
                window=9,
                overrides={"v_min": 3.0, "v_max": -3.0},
            )

    def test_the_agent_actually_builds_on_the_narrowed_support(self) -> None:
        config = qolu_config_for(
            "qolu_buia",
            observation_channels=3,
            window=3,
            overrides={"v_min": -3.0, "v_max": 3.0},
        )
        agent = QOLUAgent(config, seed=0)
        assert agent.support.v_min == pytest.approx(-3.0)
        assert agent.support.v_max == pytest.approx(3.0)
        assert agent.support.n_atoms == 51
        assert agent.support.delta_z == pytest.approx(0.12)


class TestTheGradientBudgetIsAConfigFieldDefaultingToTheCurrentBehaviour:
    def test_the_default_is_one_learn_call_per_iteration(self) -> None:
        config = QOLUConfig(observation_channels=3, observation_window=9)
        assert config.learn_steps_per_iteration == 1
        assert (
            qolu_config_for(
                "qolu_buia", observation_channels=3, window=9
            ).learn_steps_per_iteration
            == 1
        )

    def test_a_non_positive_budget_is_refused(self) -> None:
        with pytest.raises(
            ValueError, match="learn_steps_per_iteration must be positive"
        ):
            QOLUConfig(
                observation_channels=3,
                observation_window=9,
                learn_steps_per_iteration=0,
            )

    def test_it_is_reachable_through_the_same_override_route(self) -> None:
        config = qolu_config_for(
            "qolu_buia",
            observation_channels=3,
            window=9,
            overrides={"learn_steps_per_iteration": 4},
        )
        assert config.learn_steps_per_iteration == 4

    def test_a_checkpoint_written_before_the_field_existed_still_loads(self) -> None:
        """The field is additive, so a payload without it takes the default of 1."""
        from dataclasses import asdict

        from chapter4_qolu.agent.checkpoint import config_from_payload

        stored = asdict(QOLUConfig(observation_channels=3, observation_window=3))
        stored.pop("learn_steps_per_iteration")
        config = config_from_payload({"config": stored})
        assert config.learn_steps_per_iteration == 1


class TestTheTrainingLoopsHonourTheBudget:
    """One ``learn()`` per iteration is the default, and a budget is honoured."""

    @staticmethod
    def _tiny_env():
        from chapter4_qolu.tests.test_joint_loop import make_env

        return make_env(samples_per_iteration=4, max_conversions_per_iteration=1)

    def _agent(self, env, learn_steps: int) -> QOLUAgent:
        config = qolu_config_for(
            "qolu_buia",
            observation_channels=env.observation_channels,
            window=env.config.observation_window,
            overrides={
                "learning_starts": 4,
                "batch_size": 4,
                "replay_capacity": 200,
                "n_step": 2,
                "learn_steps_per_iteration": learn_steps,
                "target_sync_interval": 1_000,
            },
        )
        return QOLUAgent(config, seed=0)

    @pytest.mark.parametrize("learn_steps", (1, 3))
    def test_train_joint_fires_the_budget_each_iteration(
        self, learn_steps: int
    ) -> None:
        from chapter4_qolu.runner import train_joint

        env = self._tiny_env()
        agent = self._agent(env, learn_steps)
        train_joint(env, [agent], None, iterations=6, seed=0)
        # The buffer needs an iteration or two to reach ``learning_starts``, so
        # the count is not ``6 * learn_steps``; what must hold is that it is a
        # multiple of the budget and rises with it.
        assert agent.gradient_steps > 0
        assert agent.gradient_steps % learn_steps == 0

    def test_train_joint_multiplies_the_gradient_count(self) -> None:
        from chapter4_qolu.runner import train_joint

        counts = {}
        for learn_steps in (1, 4):
            env = self._tiny_env()
            agent = self._agent(env, learn_steps)
            train_joint(env, [agent], None, iterations=6, seed=0)
            counts[learn_steps] = agent.gradient_steps
        assert counts[4] == 4 * counts[1]

    def test_train_qolu_honours_it_too(self) -> None:
        from chapter4_qolu.runner import train_qolu

        counts = {}
        for learn_steps in (1, 4):
            env = self._tiny_env()
            agent = self._agent(env, learn_steps)
            train_qolu(env, agent, iterations=6, seed=0)
            counts[learn_steps] = agent.gradient_steps
        assert counts[1] > 0
        assert counts[4] == 4 * counts[1]


class TestTheDiagnosticsSettingsCarryThem:
    def test_the_defaults_are_none_meaning_the_package_defaults(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        assert tiny_settings.v_min is None
        assert tiny_settings.v_max is None
        assert tiny_settings.n_atoms is None
        assert tiny_settings.learn_steps_per_iteration is None
        assert tiny_settings.qolu_overrides == {}

    def test_values_become_qolu_overrides(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        settings = replace(
            tiny_settings,
            v_min=-3.0,
            v_max=3.0,
            n_atoms=51,
            learn_steps_per_iteration=4,
            target_sync_interval=25,
        )
        assert settings.qolu_overrides == {
            "target_sync_interval": 25,
            "n_atoms": 51,
            "v_min": -3.0,
            "v_max": 3.0,
            "learn_steps_per_iteration": 4,
        }

    def test_one_end_of_the_support_may_be_moved_alone(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        assert replace(tiny_settings, v_max=3.0).qolu_overrides == {"v_max": 3.0}

    def test_an_inverted_support_is_refused_by_the_settings(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        with pytest.raises(ValueError, match="v_max must be greater than v_min"):
            replace(tiny_settings, v_min=3.0, v_max=-3.0)

    def test_a_non_positive_budget_is_refused_by_the_settings(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        with pytest.raises(
            ValueError, match="learn_steps_per_iteration must be positive"
        ):
            replace(tiny_settings, learn_steps_per_iteration=0)

    def test_too_few_atoms_is_refused_by_the_settings(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        with pytest.raises(ValueError, match="n_atoms must be at least two"):
            replace(tiny_settings, n_atoms=1)

    def test_the_record_states_what_was_asked_for(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        record = tiny_settings.as_record()
        assert record["v_min"] is None
        assert record["v_max"] is None
        assert record["n_atoms"] is None
        assert record["learn_steps_per_iteration"] is None
        moved = replace(
            tiny_settings,
            v_min=-3.0,
            v_max=3.0,
            n_atoms=51,
            learn_steps_per_iteration=4,
        ).as_record()
        assert (moved["v_min"], moved["v_max"], moved["n_atoms"]) == (-3.0, 3.0, 51)
        assert moved["learn_steps_per_iteration"] == 4

    def test_a_sweep_cell_inherits_them(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        base = replace(
            tiny_settings, v_min=-3.0, v_max=3.0, learn_steps_per_iteration=4
        )
        cell = cell_settings(base, scenario=2, seed=1)
        assert (cell.v_min, cell.v_max) == (-3.0, 3.0)
        assert cell.learn_steps_per_iteration == 4


class TestTheCommandLineOffersThem:
    @staticmethod
    def _options() -> set[str]:
        return {
            action.option_strings[0]
            for action in build_parser()._actions
            if action.option_strings
        }

    @pytest.mark.parametrize(
        "flag", ("--v-min", "--v-max", "--n-atoms", "--learn-steps-per-iteration")
    )
    def test_the_flag_exists(self, flag: str) -> None:
        assert flag in self._options()

    def test_they_default_to_the_package_defaults(
        self, tiny_class_raster: Path
    ) -> None:
        settings = settings_from(
            build_parser().parse_args(["--raster", str(tiny_class_raster)])
        )
        assert settings.v_min is None
        assert settings.v_max is None
        assert settings.n_atoms is None
        assert settings.learn_steps_per_iteration is None

    def test_they_reach_the_settings(self, tiny_class_raster: Path) -> None:
        settings = settings_from(
            build_parser().parse_args(
                [
                    "--raster",
                    str(tiny_class_raster),
                    "--v-min",
                    "-3",
                    "--v-max",
                    "3",
                    "--n-atoms",
                    "51",
                    "--learn-steps-per-iteration",
                    "4",
                ]
            )
        )
        assert (settings.v_min, settings.v_max, settings.n_atoms) == (-3.0, 3.0, 51)
        assert settings.learn_steps_per_iteration == 4


class TestTheLearnerRecordSaysWhatTheRunUsed:
    def test_it_reports_the_support_and_the_budget(self) -> None:
        config = qolu_config_for(
            "qolu_buia",
            observation_channels=3,
            window=3,
            overrides={"v_min": -3.0, "v_max": 3.0, "learn_steps_per_iteration": 4},
        )
        record = learner_record([QOLUAgent(config, seed=0)])
        assert record["learn_steps_per_iteration"] == 4
        assert record["support"] == {
            "n_atoms": 51,
            "v_min": -3.0,
            "v_max": 3.0,
            "delta_z": pytest.approx(0.12),
        }

    def test_it_reports_the_published_support_by_default(self) -> None:
        config = qolu_config_for("qolu_buia", observation_channels=3, window=3)
        record = learner_record([QOLUAgent(config, seed=0)])
        assert record["learn_steps_per_iteration"] == 1
        assert record["support"]["v_min"] == -200.0
        assert record["support"]["v_max"] == 200.0
        assert record["support"]["delta_z"] == pytest.approx(8.0)

    def test_stakeholders_that_disagree_are_refused(self) -> None:
        wide = qolu_config_for("qolu_buia", observation_channels=3, window=3)
        narrow = qolu_config_for(
            "qolu_buia",
            observation_channels=3,
            window=3,
            overrides={"v_min": -3.0, "v_max": 3.0},
        )
        with pytest.raises(ValueError, match="atom support"):
            learner_record([QOLUAgent(wide, seed=0), QOLUAgent(narrow, seed=1)])


class TestEndToEnd:
    def test_the_summary_records_the_support_the_run_used(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        settings = replace(
            tiny_settings, tag="narrow", v_min=-3.0, v_max=3.0, n_atoms=51
        )
        summary = run_distribution_diagnostics(settings)
        assert summary["run"]["v_min"] == -3.0
        assert summary["run"]["v_max"] == 3.0
        assert summary["learner"]["support"]["v_min"] == -3.0
        support = summary["pooled"]["targets"]["n_step_target"]["support"]
        assert support["v_min"] == pytest.approx(-3.0)
        assert support["delta_z"] == pytest.approx(0.12)

    def test_the_published_support_is_what_an_unasked_run_uses(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        summary = run_distribution_diagnostics(tiny_settings)
        assert summary["learner"]["support"]["v_min"] == -200.0
        assert summary["learner"]["learn_steps_per_iteration"] == 1

    def test_a_bigger_budget_takes_more_gradient_steps(
        self, tiny_settings: DiagnosticsSettings
    ) -> None:
        one = run_distribution_diagnostics(replace(tiny_settings, tag="one"))
        four = run_distribution_diagnostics(
            replace(tiny_settings, tag="four", learn_steps_per_iteration=4)
        )
        assert four["learner"]["learn_steps_per_iteration"] == 4
        assert four["learner"]["gradient_steps"][0] == (
            4 * one["learner"]["gradient_steps"][0]
        )
