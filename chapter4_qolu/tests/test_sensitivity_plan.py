"""What the sensitivity sweep consists of, and what it costs."""

from __future__ import annotations

from pathlib import Path

import pytest

from chapter4_qolu.methods import QOLU_METHOD
from chapter4_qolu.sensitivity_plan import (
    CAPS,
    DEFAULT_ENVIRONMENTS,
    PMADDPG_VARIANTS,
    REWARD_MODES,
    SensitivityConfiguration,
    SensitivitySettings,
    StageBChoice,
    configurations,
    estimate_stage_seconds,
    sensitivity_paths,
)
from chapter4_qolu.sensitivity_stages import (
    stage_a_campaigns,
    stage_a_runs,
    stage_b_campaigns,
)
from chapter4_qolu.window_scan import ScanWindow


@pytest.fixture()
def settings(tmp_path: Path) -> SensitivitySettings:
    return SensitivitySettings(
        raster=tmp_path / "raster.img",
        tag="demo",
        output_dir=tmp_path / "results",
    )


class TestConfigurationGrid:
    def test_the_documented_axes_are_the_defaults(self) -> None:
        assert CAPS == (1, 3, 10)
        assert REWARD_MODES == ("per_parcel", "objective", "team")
        assert PMADDPG_VARIANTS == ("pmaddpg", "pmaddpg_reg_only")
        assert DEFAULT_ENVIRONMENTS == ("SE_region", "E495_N140", "E495_N165")

    def test_there_are_two_hundred_and_sixteen_configurations(self) -> None:
        assert len(configurations()) == 3 * 3 * 3 * 2 * 2 * 2

    def test_configurations_are_unique_and_labelled(self) -> None:
        labels = [item.label for item in configurations()]
        assert len(set(labels)) == len(labels)

    def test_a_configuration_names_its_four_methods(self) -> None:
        item = SensitivityConfiguration(
            environment="E495_N140",
            cap=3,
            reward_mode="team",
            pmaddpg="pmaddpg_reg_only",
            mitm="mitm_learned",
        )
        assert item.methods == (
            QOLU_METHOD,
            "pmaddpg_reg_only",
            "ltos",
            "mitm_learned",
        )

    def test_the_scale_carries_the_cap_and_the_extent(self) -> None:
        item = SensitivityConfiguration(
            environment="E495_N140", cap=3, reward_mode="per_parcel"
        )
        assert item.scale.max_conversions == 3
        assert item.scale.shape == (50, 50)
        assert item.scale.label == "50x50_f1.0_cap3"

    def test_the_whole_region_environment_is_rectangular(self) -> None:
        item = SensitivityConfiguration(
            environment="SE_region", cap=1, reward_mode="per_parcel"
        )
        assert item.grid_shape == (210, 240)
        assert item.scale.label == "210x240_f1.0_cap1"

    def test_the_label_names_the_environment_first(self) -> None:
        item = SensitivityConfiguration(
            environment="SE_region", cap=1, reward_mode="team"
        )
        assert item.label.startswith("SE_region_cap1_team")

    def test_an_unknown_reward_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="reward mode"):
            SensitivityConfiguration(
                environment="E495_N140", cap=1, reward_mode="nonsense"
            )

    def test_an_unknown_variant_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="variant"):
            SensitivityConfiguration(
                environment="E495_N140", cap=1, reward_mode="team", ltos="mitm"
            )


class TestStageALayout:
    def test_every_environment_gets_one_campaign_per_reward_mode_plus_baselines(
        self, settings: SensitivitySettings
    ) -> None:
        campaigns = stage_a_campaigns(settings)
        # 3 environments x (3 reward modes + 1 baseline campaign), plus one
        # documented reference run per environment.
        assert len(campaigns) == 3 * 4 + 3

    def test_the_run_count_is_thirty_nine_per_environment(
        self, settings: SensitivitySettings
    ) -> None:
        # 3 caps x (1 QOLU+BUIA + 2 P-MADDPG) x 3 reward modes = 27, plus
        # 3 caps x 4 reward-free baselines = 12, plus one reference run each.
        runs = stage_a_runs(settings)
        assert len(runs) == 3 * 40

    def test_baselines_are_run_once_per_cap_not_once_per_reward_mode(
        self, settings: SensitivitySettings
    ) -> None:
        runs = stage_a_runs(settings)
        ltos = [spec for spec in runs if spec.method == "ltos"]
        # 3 environments x 3 caps: LToS never reads a QOLU reward mode.
        assert len(ltos) == 3 * len(CAPS)

    def test_the_unregularised_variant_is_one_reference_run_per_environment(
        self, settings: SensitivitySettings
    ) -> None:
        runs = [s for s in stage_a_runs(settings) if s.method == "pmaddpg_v1"]
        assert len(runs) == len(DEFAULT_ENVIRONMENTS)
        assert {s.window_label for s in runs} == set(DEFAULT_ENVIRONMENTS)
        # Always at the published coordinates, never swept.
        assert {s.max_conversions for s in runs} == {10}
        assert {s.qolu_reward_mode for s in runs} == {"per_parcel"}

    def test_the_neural_methods_are_run_once_per_reward_mode(
        self, settings: SensitivitySettings
    ) -> None:
        runs = stage_a_runs(settings)
        qolu = [spec for spec in runs if spec.method == QOLU_METHOD]
        assert len(qolu) == 3 * len(CAPS) * len(REWARD_MODES)
        assert {spec.qolu_reward_mode for spec in qolu} == set(REWARD_MODES)

    def test_stage_a_is_scenario_one_seed_zero_and_unweighted(
        self, settings: SensitivitySettings
    ) -> None:
        for spec in stage_a_runs(settings):
            assert spec.scenario == 1
            assert spec.seed == 0
            assert spec.scale_by_scenario_weight is False
            assert spec.weighted_consensus is False

    def test_every_run_pins_its_environment(
        self, settings: SensitivitySettings
    ) -> None:
        labels = {spec.window_label for spec in stage_a_runs(settings)}
        assert labels == set(DEFAULT_ENVIRONMENTS)
        for spec in stage_a_runs(settings):
            assert spec.window_bng_km is not None

    def test_runs_land_under_the_stage_a_directory(
        self, settings: SensitivitySettings
    ) -> None:
        paths = sensitivity_paths(settings)
        for spec in stage_a_runs(settings):
            assert paths.stage_a in spec.output_path.parents

    def test_every_run_has_a_distinct_output_path(
        self, settings: SensitivitySettings
    ) -> None:
        paths = [spec.output_path for spec in stage_a_runs(settings)]
        assert len(set(paths)) == len(paths)


class TestStageBLayout:
    def _choices(self) -> list[StageBChoice]:
        return [
            StageBChoice(
                rank=1,
                configuration=SensitivityConfiguration(
                    environment="E495_N140", cap=1, reward_mode="per_parcel"
                ),
                window=ScanWindow(easting_km=495, northing_km=140),
            ),
            StageBChoice(
                rank=2,
                configuration=SensitivityConfiguration(
                    environment="SE_region",
                    cap=3,
                    reward_mode="team",
                    pmaddpg="pmaddpg_reg_only",
                    mitm="mitm_learned",
                ),
                window=ScanWindow(easting_km=495, northing_km=165),
            ),
        ]

    def test_each_choice_gets_a_plain_and_a_weighted_campaign(
        self, settings: SensitivitySettings
    ) -> None:
        campaigns = stage_b_campaigns(settings, self._choices())
        assert len(campaigns) == 4
        weighted = [c for c in campaigns if c.weighted_consensus]
        plain = [c for c in campaigns if not c.weighted_consensus]
        assert len(weighted) == len(plain) == 2
        for campaign in weighted:
            assert campaign.scale_by_scenario_weight is True

    def test_a_stage_b_campaign_runs_four_scenarios_and_three_seeds(
        self, settings: SensitivitySettings
    ) -> None:
        campaign = stage_b_campaigns(settings, self._choices())[0]
        assert campaign.scenarios == (1, 2, 3, 4)
        assert campaign.seeds == (0, 1, 2)
        assert len(campaign.methods) == 4

    def test_campaign_tags_are_unique_and_name_the_reading(
        self, settings: SensitivitySettings
    ) -> None:
        tags = [c.tag for c in stage_b_campaigns(settings, self._choices())]
        assert len(set(tags)) == len(tags)
        assert any(tag.endswith("_plain") for tag in tags)
        assert any(tag.endswith("_weighted") for tag in tags)

    def test_stage_b_campaigns_live_under_the_stage_b_directory(
        self, settings: SensitivitySettings
    ) -> None:
        paths = sensitivity_paths(settings)
        for campaign in stage_b_campaigns(settings, self._choices()):
            assert paths.stage_b in campaign.campaign_dir.parents


class TestCost:
    def test_both_stages_have_a_positive_estimate(
        self, settings: SensitivitySettings
    ) -> None:
        stage_a = estimate_stage_seconds(stage_a_runs(settings))
        assert stage_a > 0.0

    def test_the_single_thread_correction_is_applied(
        self, settings: SensitivitySettings
    ) -> None:
        from chapter4_qolu.campaign_plan import estimate_runtime_seconds
        from chapter4_qolu.scan_plan import SINGLE_THREAD_COST_FACTOR

        runs = stage_a_runs(settings)
        assert estimate_stage_seconds(runs) == pytest.approx(
            estimate_runtime_seconds(runs) * SINGLE_THREAD_COST_FACTOR
        )
