"""Tests for the three named QOLU readings and how a method key resolves one.

``qolu_buia`` is the chapter's reading, ``qolu_buia_per`` adds the opt-in
prioritised replay, and ``qolu_c51`` is the plain C51 reading.  Laid out like
the P-MADDPG variant tests, because the mechanism is the same one.
"""

from __future__ import annotations

import pytest
from torch import nn

from chapter4_qolu.agent.noisy import NoisyLinear
from chapter4_qolu.agent.prioritised_replay import PrioritisedReplayBuffer
from chapter4_qolu.agent.qolu_variants import QOLU_VARIANTS, qolu_variant_settings
from chapter4_qolu.agent.replay import NStepReplayBuffer
from chapter4_qolu.tests.test_qolu_rainbow import C51_ONLY, make_agent


class TestVariantPresets:
    def test_the_three_readings_are_registered(self) -> None:
        assert set(QOLU_VARIANTS) == {"published", "per", "c51"}

    def test_the_published_variant_changes_nothing(self) -> None:
        assert qolu_variant_settings("published") == {}

    def test_the_per_variant_only_turns_prioritisation_on(self) -> None:
        assert qolu_variant_settings("per") == {"prioritised_replay": True}

    def test_the_c51_variant_turns_both_switches_off(self) -> None:
        assert qolu_variant_settings("c51") == C51_ONLY

    def test_an_unknown_variant_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown QOLU variant"):
            qolu_variant_settings("rainbow-lite")

    def test_the_settings_are_a_fresh_copy(self) -> None:
        settings = qolu_variant_settings("c51")
        settings["noisy_nets"] = True
        assert qolu_variant_settings("c51")["noisy_nets"] is False

    def test_the_presets_build_the_agents_they_name(self) -> None:
        published = make_agent(**qolu_variant_settings("published"))
        prioritised = make_agent(**qolu_variant_settings("per"))
        c51 = make_agent(**qolu_variant_settings("c51"))
        assert type(published.replay) is NStepReplayBuffer
        assert isinstance(prioritised.replay, PrioritisedReplayBuffer)
        assert type(c51.replay) is NStepReplayBuffer
        assert isinstance(published.online_network.value_head, NoisyLinear)
        assert isinstance(prioritised.online_network.value_head, NoisyLinear)
        assert type(c51.online_network.value_head) is nn.Linear


class TestMethodRegistry:
    def test_all_three_qolu_keys_are_registered(self) -> None:
        from chapter4_qolu.methods import METHOD_KEYS, QOLU_METHOD_VARIANTS

        for key in ("qolu_buia", "qolu_buia_per", "qolu_c51"):
            assert key in METHOD_KEYS
        assert QOLU_METHOD_VARIANTS == {
            "qolu_buia": "published",
            "qolu_buia_per": "per",
            "qolu_c51": "c51",
        }

    def test_the_extra_keys_are_variants_of_the_published_system(self) -> None:
        from chapter4_qolu.methods import canonical_method, is_variant

        assert canonical_method("qolu_c51") == "qolu_buia"
        assert canonical_method("qolu_buia_per") == "qolu_buia"
        assert is_variant("qolu_c51") is True
        assert is_variant("qolu_buia_per") is True
        assert is_variant("qolu_buia") is False

    def test_each_variant_has_a_label_a_row_and_a_price(self) -> None:
        from chapter4_qolu.campaign_plan import METHOD_COST_SECONDS
        from chapter4_qolu.reporting.tables import METHOD_LABELS, PAPER_ROW_ORDER

        for key in ("qolu_c51", "qolu_buia_per"):
            assert "QOLU" in METHOD_LABELS[key]
            assert PAPER_ROW_ORDER.index(key) > PAPER_ROW_ORDER.index("qolu_buia")
            assert key in METHOD_COST_SECONDS

    def test_the_run_spec_accepts_the_variants(self) -> None:
        from pathlib import Path

        from chapter4_qolu.reporting.paths import parse_scale_config
        from chapter4_qolu.scenario_worker import RunSpec

        for key in ("qolu_buia", "qolu_buia_per", "qolu_c51"):
            spec = RunSpec(
                raster=Path("unused.tif"),
                method=key,
                scenario=1,
                seed=0,
                scale=parse_scale_config("8:1.0:10"),
                output_path=Path("unused.json"),
                iterations=2,
            )
            assert spec.method == key

    def test_the_worker_builds_the_variant_the_key_names(self) -> None:
        from chapter4_qolu.scenario_worker import qolu_config_for

        published = qolu_config_for("qolu_buia", observation_channels=4, window=5)
        prioritised = qolu_config_for("qolu_buia_per", observation_channels=4, window=5)
        c51 = qolu_config_for("qolu_c51", observation_channels=4, window=5)
        assert published.noisy_nets is True and published.prioritised_replay is False
        assert prioritised.noisy_nets is True and prioritised.prioritised_replay is True
        assert c51.noisy_nets is False and c51.prioritised_replay is False

    def test_an_unknown_key_cannot_be_built(self) -> None:
        from chapter4_qolu.scenario_worker import qolu_config_for

        with pytest.raises(ValueError):
            qolu_config_for("pmaddpg", observation_channels=4, window=5)
