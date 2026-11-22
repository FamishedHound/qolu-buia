"""Tests for the comparison against the published ranking pattern."""

from __future__ import annotations

from typing import Any

import pytest

from chapter4_qolu.reporting.aggregate import aggregate_records
from chapter4_qolu.reporting.pattern import (
    EMPHASISED_METRICS,
    PUBLISHED_RANKING,
    check_cell,
    render_pattern_check,
)
from chapter4_qolu.tests.test_reporting_aggregate import make_record


class TestEmphasisedMetrics:
    def test_scenario_one_emphasises_all_three_metrics(self) -> None:
        assert EMPHASISED_METRICS[1] == ("land_use", "woodland", "urban")

    def test_each_other_scenario_emphasises_its_own_metric(self) -> None:
        assert EMPHASISED_METRICS[2] == ("land_use",)
        assert EMPHASISED_METRICS[3] == ("woodland",)
        assert EMPHASISED_METRICS[4] == ("urban",)

    def test_published_ranking_is_qolu_then_pmaddpg_then_ltos_then_mitm(self) -> None:
        assert PUBLISHED_RANKING == ("qolu_buia", "pmaddpg", "ltos", "mitm")


class TestCheckCell:
    def test_the_published_order_agrees_on_an_emphasised_metric(self) -> None:
        cell = check_cell(2, "land_use", PUBLISHED_RANKING)
        assert cell.emphasised is True
        assert cell.reference_rank == 1
        assert cell.reference_first is True
        assert cell.baselines_in_published_order is True
        assert cell.agrees is True

    def test_reference_second_fails_an_emphasised_metric(self) -> None:
        cell = check_cell(2, "land_use", ("pmaddpg", "qolu_buia", "ltos", "mitm"))
        assert cell.emphasised is True
        assert cell.reference_rank == 2
        assert cell.reference_first is False
        assert cell.agrees is False

    def test_reference_second_is_tolerated_on_a_non_emphasised_metric(self) -> None:
        # The published tables tie QOLU+BUIA with P-MADDPG away from the
        # emphasised metric, so "top two" is the expectation there.
        cell = check_cell(2, "woodland", ("pmaddpg", "qolu_buia", "ltos", "mitm"))
        assert cell.emphasised is False
        assert cell.reference_first is False
        assert cell.agrees is True

    def test_reference_third_fails_even_a_non_emphasised_metric(self) -> None:
        cell = check_cell(2, "woodland", ("pmaddpg", "ltos", "qolu_buia", "mitm"))
        assert cell.agrees is False

    def test_baselines_out_of_order_fail(self) -> None:
        cell = check_cell(1, "urban", ("qolu_buia", "ltos", "pmaddpg", "mitm"))
        assert cell.reference_first is True
        assert cell.baselines_in_published_order is False
        assert cell.agrees is False

    def test_only_compares_methods_that_are_present(self) -> None:
        cell = check_cell(1, "urban", ("qolu_buia", "mitm"))
        assert cell.baselines_in_published_order is True
        assert cell.agrees is True

    def test_a_missing_reference_method_cannot_agree(self) -> None:
        cell = check_cell(1, "urban", ("pmaddpg", "ltos", "mitm"))
        assert cell.reference_rank is None
        assert cell.agrees is False

    def test_verdict_is_a_short_human_readable_string(self) -> None:
        assert check_cell(2, "land_use", PUBLISHED_RANKING).verdict
        assert check_cell(2, "land_use", ("mitm",)).verdict


class TestRenderPatternCheck:
    @pytest.fixture()
    def summary(self) -> dict[str, Any]:
        series = {
            "qolu_buia": [0.80, 0.86, 0.83],
            "pmaddpg": [0.70, 0.71, 0.75],
            "ltos": [0.60, 0.65, 0.61],
            "mitm": [0.50, 0.58, 0.52],
        }
        records = [
            make_record(
                method,
                seed,
                {"land_use": value, "woodland": value, "urban": value},
                scenario=scenario,
            )
            for scenario in (1, 2, 3, 4)
            for method, values in series.items()
            for seed, value in enumerate(values)
        ]
        return aggregate_records(records, metadata={"tag": "unit"})

    def test_reports_one_row_per_scenario_and_metric(
        self, summary: dict[str, Any]
    ) -> None:
        text = render_pattern_check(summary)
        rows = [line for line in text.splitlines() if line.startswith("| Scenario ")]
        assert len(rows) == 4 * 3

    def test_states_that_only_the_ordering_is_compared(
        self, summary: dict[str, Any]
    ) -> None:
        text = render_pattern_check(summary)
        assert "ordering" in text
        assert "0.05" in text

    def test_describes_the_published_pattern_it_is_checking(
        self, summary: dict[str, Any]
    ) -> None:
        text = render_pattern_check(summary)
        assert "P-MADDPG" in text and "LToS" in text and "MitM" in text

    def test_counts_the_agreeing_cells(self, summary: dict[str, Any]) -> None:
        text = render_pattern_check(summary)
        assert "12 of 12" in text

    def test_flags_a_disagreeing_cell(self) -> None:
        series = {
            "qolu_buia": [0.10, 0.12, 0.11],
            "pmaddpg": [0.70, 0.71, 0.75],
            "ltos": [0.60, 0.65, 0.61],
            "mitm": [0.50, 0.58, 0.52],
        }
        records = [
            make_record(
                method,
                seed,
                {"land_use": value, "woodland": value, "urban": value},
                scenario=1,
            )
            for method, values in series.items()
            for seed, value in enumerate(values)
        ]
        text = render_pattern_check(aggregate_records(records))
        assert "0 of 3" in text
        assert "no" in text.lower()


class TestRegionCaveat:
    """The region sentence has to describe what the campaign actually ran on."""

    def _summary(self, campaign: dict) -> dict:
        return {
            "reference_method": "qolu_buia",
            "campaign": campaign,
            "configurations": [],
        }

    def test_names_the_pinned_window_and_the_scan_that_selected_it(self) -> None:
        text = render_pattern_check(
            self._summary(
                {"tag": "E470_N190", "raster": "gb.img", "window_label": "E470_N190"}
            )
        )
        assert "E470_N190" in text
        assert "the window scan selected" in text
        assert "Northern-Ireland placeholder" not in text

    def test_says_the_loader_picked_the_extent_when_no_window_was_pinned(self) -> None:
        text = render_pattern_check(self._summary({"tag": "main", "raster": "gb.img"}))
        assert "the extent the loader selected" in text
        assert "`gb.img`" in text

    def test_falls_back_when_no_raster_is_recorded(self) -> None:
        text = render_pattern_check(self._summary({"tag": "main"}))
        assert "whatever raster this campaign was handed" in text


class TestVariantBaselinesAreNotSkipped:
    """A variant baseline must not make the order check vacuous.

    Filtering the ranking by literal membership of ``PUBLISHED_RANKING`` would
    leave a campaign running ``mitm_learned`` and ``pmaddpg_reg_only`` with
    only ``ltos`` in the expected order, the comparison would become
    ``["ltos"] == ["ltos"]``, and every cell would be credited with baselines
    "in the published order" however they were actually ordered.
    """

    def test_a_wrong_variant_order_is_caught(self) -> None:
        # ltos ahead of P-MADDPG is *not* the published baseline order.
        cell = check_cell(
            1,
            "land_use",
            ("qolu_buia", "ltos", "pmaddpg_reg_only", "mitm_learned"),
        )
        assert cell.baselines_in_published_order is False
        assert cell.agrees is False

    def test_the_right_variant_order_still_passes(self) -> None:
        cell = check_cell(
            1,
            "land_use",
            ("qolu_buia", "pmaddpg_reg_only", "fixed_ltos", "mitm_learned"),
        )
        assert cell.baselines_in_published_order is True
        assert cell.agrees is True

    def test_variants_are_reported_under_their_published_names(self) -> None:
        from chapter4_qolu.reporting.pattern import _published_text

        text = _published_text(
            ("qolu_buia", "pmaddpg_reg_only", "fixed_ltos", "mitm_learned")
        )
        assert text == "QOLU+BUIA > P-MADDPG > LToS > MitM"

    def test_the_published_four_are_unaffected(self) -> None:
        good = check_cell(1, "urban", ("qolu_buia", "pmaddpg", "ltos", "mitm"))
        bad = check_cell(1, "urban", ("qolu_buia", "ltos", "pmaddpg", "mitm"))
        assert good.agrees is True
        assert bad.agrees is False

    def test_a_ranking_made_only_of_variant_keys_is_still_checked(self) -> None:
        """Every name in the ranking may be a variant, including the reference."""
        cell = check_cell(
            2,
            "land_use",
            ("qolu_c51", "pmaddpg_v1", "fixed_ltos", "mitm_learned"),
        )
        assert cell.reference_rank == 1
        assert cell.baselines_in_published_order is True
        assert cell.agrees is True


class TestQOLUVariantsAreRecognised:
    """The QOLU *reading* is a variant key too.

    ``qolu_c51`` and ``qolu_buia_per`` are the plain C51 and prioritised
    readings of the published system.  They must be read as QOLU + BUIA
    wherever a ranking is compared with the published tables; otherwise the
    reference method "did not run in this cell" and the whole configuration is
    scored as a disagreement.
    """

    def test_the_qolu_variants_stand_in_for_the_published_system(self) -> None:
        for key in ("qolu_c51", "qolu_buia_per"):
            cell = check_cell(3, "woodland", (key, "pmaddpg", "ltos", "mitm"))
            assert cell.reference_rank == 1, key
            assert cell.reference_first is True, key
            assert cell.agrees is True, key

    def test_a_qolu_variant_that_ranks_last_still_disagrees(self) -> None:
        cell = check_cell(3, "woodland", ("pmaddpg", "ltos", "mitm", "qolu_c51"))
        assert cell.reference_rank == 4
        assert cell.agrees is False

    def test_the_published_column_names_the_parent_method(self) -> None:
        from chapter4_qolu.reporting.pattern import _published_text

        text = _published_text(("qolu_c51", "pmaddpg_reg_only", "fixed_ltos", "mitm"))
        assert text == "QOLU+BUIA > P-MADDPG > LToS > MitM"

    def test_the_rebuild_column_keeps_the_variant_label(self) -> None:
        from chapter4_qolu.reporting.pattern import _ranking_text

        assert "C51" in _ranking_text(("qolu_c51",))


class TestTheAblationIsLeftOut:
    """``no_regulator`` has no published row, so it must not move any rank."""

    def test_it_does_not_push_the_reference_down(self) -> None:
        cell = check_cell(
            2, "land_use", ("no_regulator", "qolu_buia", "pmaddpg", "ltos", "mitm")
        )
        assert cell.reference_rank == 1
        assert cell.agrees is True
        assert "no_regulator" not in cell.ranking

    def test_it_is_not_listed_in_the_rebuild_order(self) -> None:
        records = [
            make_record(method, seed, {"land_use": value})
            for seed in range(3)
            for method, value in (
                ("no_regulator", 0.99),
                ("qolu_buia", 0.9),
                ("pmaddpg", 0.8),
                ("ltos", 0.7),
                ("mitm", 0.6),
            )
        ]
        text = render_pattern_check(aggregate_records(records))
        assert "No regulator" not in text
