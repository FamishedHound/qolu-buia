"""The +/-20 % hyper-parameter perturbations and the plan they generate."""

from __future__ import annotations

from pathlib import Path

import pytest

from chapter4_qolu.config import EnvConfig, QOLUConfig
from chapter4_qolu.hparam_plan import (
    HParamSettings,
    all_runs,
    baseline_campaign,
    baseline_runs,
    effective_perturbations,
    estimate_seconds,
    hparam_paths,
    perturbation_campaign,
    perturbation_runs,
    settings_perturbations,
)
from chapter4_qolu.hparam_sensitivity import (
    KEY_HYPERPARAMETERS,
    PERTURBATION_FRACTION,
    HyperParameter,
    kendall_tau,
    perturb,
    perturbations,
    resolve_hyperparameters,
)
from chapter4_qolu.hparam_worker import HParamRunSpec, as_hparam_spec
from chapter4_qolu.methods import PUBLISHED_METHODS, QOLU_METHOD


@pytest.fixture()
def settings(tmp_path: Path) -> HParamSettings:
    return HParamSettings(
        raster=tmp_path / "raster.img",
        tag="demo",
        output_dir=tmp_path / "results",
    )


def _named(name: str) -> HyperParameter:
    return resolve_hyperparameters([name])[0]


class TestTheKeyHyperParameters:
    def test_the_thesis_fraction_is_twenty_percent(self) -> None:
        assert PERTURBATION_FRACTION == pytest.approx(0.20)

    def test_the_documented_eleven_are_the_constant(self) -> None:
        assert tuple(item.name for item in KEY_HYPERPARAMETERS) == (
            "learning_rate",
            "gamma",
            "batch_size",
            "replay_capacity",
            "n_step",
            "target_sync_interval",
            "epsilon_start",
            "epsilon_end",
            "epsilon_decay_steps",
            "samples_per_iteration",
            "observation_window",
        )

    def test_names_are_unique(self) -> None:
        names = [item.name for item in KEY_HYPERPARAMETERS]
        assert len(set(names)) == len(names)

    def test_every_baseline_is_the_package_default(self) -> None:
        env = EnvConfig()
        qolu = QOLUConfig(observation_channels=1, observation_window=9)
        for item in KEY_HYPERPARAMETERS:
            source = env if item.target == "environment" else qolu
            assert item.baseline == pytest.approx(getattr(source, item.name)), item.name

    def test_each_hyperparameter_says_which_config_it_lives_on(self) -> None:
        targets = {item.name: item.target for item in KEY_HYPERPARAMETERS}
        assert targets["learning_rate"] == "qolu"
        assert targets["samples_per_iteration"] == "environment"
        assert targets["observation_window"] == "environment"

    def test_every_hyperparameter_records_why_it_is_included(self) -> None:
        for item in KEY_HYPERPARAMETERS:
            assert item.source in {"published", "implementation"}
            assert item.note

    def test_resolving_an_unknown_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown hyper-parameter"):
            resolve_hyperparameters(["not_a_hyperparameter"])

    def test_resolving_keeps_the_requested_order(self) -> None:
        chosen = resolve_hyperparameters(["gamma", "learning_rate"])
        assert [item.name for item in chosen] == ["gamma", "learning_rate"]

    def test_resolving_an_empty_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            resolve_hyperparameters([])


class TestPerturbationArithmetic:
    def test_a_continuous_hyperparameter_moves_by_exactly_twenty_percent(self) -> None:
        down = perturb(_named("learning_rate"), -1)
        up = perturb(_named("learning_rate"), +1)
        assert down.value == pytest.approx(2.5e-4 * 0.8)
        assert up.value == pytest.approx(2.5e-4 * 1.2)
        assert not down.rounded and not up.rounded
        assert not down.clamped and not up.clamped

    def test_an_integer_hyperparameter_is_rounded_and_says_so(self) -> None:
        up = perturb(_named("batch_size"), +1)
        down = perturb(_named("batch_size"), -1)
        assert up.value == 307 and isinstance(up.value, int)
        assert down.value == 205 and isinstance(down.value, int)
        assert up.rounded and down.rounded
        assert up.requested == pytest.approx(307.2)
        assert down.requested == pytest.approx(204.8)

    def test_rounding_is_half_away_from_zero(self) -> None:
        item = HyperParameter(
            name="demo",
            target="qolu",
            baseline=2.5,
            integral=True,
            source="implementation",
            note="test only",
        )
        # 2.5 * 1.2 = 3.0 exactly; 2.5 * 0.8 = 2.0 exactly, so use a half case.
        half = HyperParameter(
            name="demo",
            target="qolu",
            baseline=1.25,
            integral=True,
            source="implementation",
            note="test only",
        )
        assert perturb(item, +1).value == 3
        assert perturb(half, +1).value == 2  # 1.5 -> 2, not 1

    def test_n_step_rounds_in_both_directions(self) -> None:
        assert perturb(_named("n_step"), -1).value == 2  # 2.4
        assert perturb(_named("n_step"), +1).value == 4  # 3.6

    def test_the_observation_window_stays_odd(self) -> None:
        assert perturb(_named("observation_window"), -1).value == 7
        assert perturb(_named("observation_window"), +1).value == 11

    def test_an_even_result_is_pushed_away_from_the_baseline(self) -> None:
        item = HyperParameter(
            name="demo",
            target="environment",
            baseline=5,
            integral=True,
            odd=True,
            source="implementation",
            note="test only",
        )
        assert perturb(item, -1).value == 3  # 4.0 is even -> step down
        assert perturb(item, +1).value == 7  # 6.0 is even -> step up

    def test_a_bounded_hyperparameter_is_clamped_and_says_so(self) -> None:
        up = perturb(_named("gamma"), +1)
        assert up.requested == pytest.approx(0.99 * 1.2)
        assert up.value == pytest.approx(1.0)
        assert up.clamped
        assert up.applied_fraction == pytest.approx((1.0 - 0.99) / 0.99)

    def test_a_clamp_back_onto_the_baseline_is_a_no_op(self) -> None:
        up = perturb(_named("epsilon_start"), +1)
        assert up.clamped
        assert up.is_no_op
        down = perturb(_named("epsilon_start"), -1)
        assert down.value == pytest.approx(0.8)
        assert not down.is_no_op

    def test_a_rounding_collapse_is_also_a_no_op(self) -> None:
        item = HyperParameter(
            name="demo",
            target="qolu",
            baseline=2,
            integral=True,
            source="implementation",
            note="test only",
        )
        assert perturb(item, +1).is_no_op  # 2.4 -> 2
        assert perturb(item, -1).is_no_op  # 1.6 -> 2

    def test_the_direction_must_be_plus_or_minus_one(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            perturb(_named("gamma"), 0)

    def test_labels_are_unique_and_name_the_direction(self) -> None:
        items = perturbations(KEY_HYPERPARAMETERS)
        labels = [item.label for item in items]
        assert len(set(labels)) == len(labels)
        assert "gamma_minus" in labels
        assert "gamma_plus" in labels

    def test_there_are_two_perturbations_per_hyperparameter(self) -> None:
        assert len(perturbations(KEY_HYPERPARAMETERS)) == 2 * len(KEY_HYPERPARAMETERS)

    def test_the_record_carries_everything_the_report_needs(self) -> None:
        record = perturb(_named("batch_size"), +1).as_record()
        assert record["hyperparameter"] == "batch_size"
        assert record["direction"] == "plus"
        assert record["baseline"] == 256
        assert record["requested"] == pytest.approx(307.2)
        assert record["value"] == 307
        assert record["rounded"] is True
        assert record["clamped"] is False
        assert record["no_op"] is False
        assert record["target"] == "qolu"

    def test_the_fraction_is_overridable(self) -> None:
        item = perturb(_named("learning_rate"), +1, fraction=0.5)
        assert item.value == pytest.approx(2.5e-4 * 1.5)
        assert item.fraction == pytest.approx(0.5)


class TestKendallTau:
    def test_an_identical_ranking_scores_one(self) -> None:
        order = ("a", "b", "c", "d")
        assert kendall_tau(order, order) == pytest.approx(1.0)

    def test_a_reversed_ranking_scores_minus_one(self) -> None:
        assert kendall_tau(("a", "b", "c"), ("c", "b", "a")) == pytest.approx(-1.0)

    def test_one_adjacent_swap_of_four(self) -> None:
        # 6 pairs, one discordant: (6 - 2 * 1) / 6
        value = kendall_tau(("a", "b", "c", "d"), ("b", "a", "c", "d"))
        assert value == pytest.approx((6 - 2) / 6)

    def test_different_members_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same methods"):
            kendall_tau(("a", "b"), ("a", "c"))

    def test_a_single_method_has_no_pairs(self) -> None:
        assert kendall_tau(("a",), ("a",)) == pytest.approx(1.0)


class TestPlan:
    def test_the_defaults_are_the_published_comparison(
        self, settings: HParamSettings
    ) -> None:
        assert settings.methods == PUBLISHED_METHODS
        assert settings.scenarios == (1,)
        assert settings.seeds == (0, 1, 2)
        assert settings.scale.label == "50x50_f1.0_cap10"

    def test_at_least_three_seeds_are_required(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="three seeds"):
            HParamSettings(raster=tmp_path / "r.img", seeds=(0, 1))

    def test_the_baseline_campaign_runs_every_method(
        self, settings: HParamSettings
    ) -> None:
        campaign = baseline_campaign(settings)
        assert campaign.methods == PUBLISHED_METHODS
        assert campaign.tag == "baseline"
        assert len(baseline_runs(settings)) == 4 * 1 * 3

    def test_an_agent_side_perturbation_only_reruns_the_qolu_learner(
        self, settings: HParamSettings
    ) -> None:
        item = perturb(_named("learning_rate"), +1)
        campaign = perturbation_campaign(settings, item)
        assert campaign.methods == (QOLU_METHOD,)
        assert len(perturbation_runs(settings, item)) == 3

    def test_an_environment_side_perturbation_reruns_all_four(
        self, settings: HParamSettings
    ) -> None:
        item = perturb(_named("samples_per_iteration"), -1)
        campaign = perturbation_campaign(settings, item)
        assert campaign.methods == PUBLISHED_METHODS
        assert len(perturbation_runs(settings, item)) == 4 * 3

    def test_a_no_op_perturbation_plans_no_runs_at_all(
        self, settings: HParamSettings
    ) -> None:
        item = perturb(_named("epsilon_start"), +1)
        assert item.is_no_op
        assert perturbation_runs(settings, item) == ()

    def test_effective_perturbations_drop_the_no_ops(
        self, settings: HParamSettings
    ) -> None:
        every = settings_perturbations(settings)
        effective = effective_perturbations(settings)
        assert len(every) == 22
        assert len(effective) == 21
        assert all(not item.is_no_op for item in effective)

    def test_every_run_has_a_distinct_output_path(
        self, settings: HParamSettings
    ) -> None:
        paths = [spec.output_path for spec in all_runs(settings)]
        assert len(set(paths)) == len(paths)

    def test_the_whole_plan_is_the_baseline_plus_every_perturbation(
        self, settings: HParamSettings
    ) -> None:
        expected = 4 * 3 + 9 * 2 * 3 - 3 + 2 * 2 * 4 * 3
        assert len(all_runs(settings)) == expected

    def test_runs_land_under_the_runs_directory(self, settings: HParamSettings) -> None:
        paths = hparam_paths(settings)
        for spec in all_runs(settings):
            assert paths.runs in spec.output_path.parents

    def test_every_run_carries_its_perturbation(self, settings: HParamSettings) -> None:
        for spec in all_runs(settings):
            assert isinstance(spec, HParamRunSpec)
        labels = {
            spec.perturbation.label
            for spec in all_runs(settings)
            if spec.perturbation is not None
        }
        assert "gamma_plus" in labels

    def test_baseline_runs_carry_no_perturbation(
        self, settings: HParamSettings
    ) -> None:
        assert all(spec.perturbation is None for spec in baseline_runs(settings))

    def test_the_estimate_is_positive(self, settings: HParamSettings) -> None:
        assert estimate_seconds(all_runs(settings)) > 0.0

    def test_the_hyperparameter_subset_is_honoured(self, tmp_path: Path) -> None:
        chosen = HParamSettings(
            raster=tmp_path / "r.img",
            hyperparameters=("gamma", "n_step"),
            output_dir=tmp_path / "results",
        )
        assert len(settings_perturbations(chosen)) == 4


class TestRunSpecOverrides:
    def _spec(
        self, settings: HParamSettings, name: str, direction: int
    ) -> HParamRunSpec:
        item = perturb(_named(name), direction)
        specs = perturbation_runs(settings, item)
        return specs[0]

    def test_an_environment_perturbation_reaches_the_env_config(
        self, settings: HParamSettings
    ) -> None:
        spec = self._spec(settings, "samples_per_iteration", -1)
        assert spec.env_config().samples_per_iteration == 40

    def test_the_observation_window_reaches_the_env_config(
        self, settings: HParamSettings
    ) -> None:
        spec = self._spec(settings, "observation_window", +1)
        assert spec.env_config().observation_window == 11

    def test_an_agent_perturbation_leaves_the_env_config_alone(
        self, settings: HParamSettings
    ) -> None:
        spec = self._spec(settings, "learning_rate", +1)
        default = EnvConfig()
        assert spec.env_config().samples_per_iteration == default.samples_per_iteration
        assert spec.env_config().observation_window == default.observation_window
        assert spec.qolu_overrides == {"learning_rate": pytest.approx(3.0e-4)}

    def test_the_perturbed_agent_config_is_valid(
        self, settings: HParamSettings
    ) -> None:
        for name in ("learning_rate", "gamma", "batch_size", "n_step"):
            for direction in (-1, +1):
                spec = self._spec(settings, name, direction)
                config = QOLUConfig(
                    observation_channels=11,
                    observation_window=9,
                    **spec.qolu_overrides,
                )
                assert getattr(config, name) == spec.perturbation.value

    def test_a_baseline_spec_round_trips_without_a_perturbation(
        self, settings: HParamSettings
    ) -> None:
        spec = baseline_runs(settings)[0]
        assert spec.qolu_overrides == {}
        assert spec.env_config() == spec.env_config()

    def test_conversion_preserves_every_run_spec_field(
        self, settings: HParamSettings
    ) -> None:
        spec = baseline_runs(settings)[0]
        again = as_hparam_spec(spec, None)
        assert again.output_path == spec.output_path
        assert again.method == spec.method
        assert again.scale == spec.scale
