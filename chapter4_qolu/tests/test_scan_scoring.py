"""Tests for the stage-1 window score of the window scan."""

from __future__ import annotations

import pytest

from chapter4_qolu.scan_scoring import (
    PUBLISHED_SCENARIO1_ORDER,
    count_pattern_matches,
    kendall_tau,
    kendall_tau_on_values,
    score_window,
)

_PERFECT = {
    "land_use": {"qolu_buia": 0.9, "pmaddpg": 0.7, "ltos": 0.5, "mitm": 0.3},
    "woodland": {"qolu_buia": 0.8, "pmaddpg": 0.6, "ltos": 0.4, "mitm": 0.2},
    "urban": {"qolu_buia": 0.7, "pmaddpg": 0.5, "ltos": 0.3, "mitm": 0.1},
}

_REVERSED = {
    metric: {method: 1.0 - value for method, value in values.items()}
    for metric, values in _PERFECT.items()
}

_FLAT = {
    metric: {name: 0.0 for name in PUBLISHED_SCENARIO1_ORDER}
    for metric in ("land_use", "woodland", "urban")
}


class TestKendallTau:
    def test_identical_rankings_score_one(self) -> None:
        order = PUBLISHED_SCENARIO1_ORDER
        assert kendall_tau(order, order) == pytest.approx(1.0)

    def test_reversed_rankings_score_minus_one(self) -> None:
        order = PUBLISHED_SCENARIO1_ORDER
        assert kendall_tau(tuple(reversed(order)), order) == pytest.approx(-1.0)

    def test_one_adjacent_swap_costs_one_of_six_pairs(self) -> None:
        swapped = ("pmaddpg", "qolu_buia", "ltos", "mitm")
        assert kendall_tau(swapped, PUBLISHED_SCENARIO1_ORDER) == pytest.approx(
            4.0 / 6.0
        )

    def test_rejects_rankings_over_different_items(self) -> None:
        with pytest.raises(ValueError):
            kendall_tau(("a", "b"), ("a", "c"))

    def test_tied_values_contribute_nothing(self) -> None:
        # Three of the four methods score identically; only the pairs that
        # separate two methods may count for or against the published order.
        values = {"qolu_buia": 0.2, "ltos": 0.2, "mitm": 0.2, "pmaddpg": 0.5}
        # Concordant: none of the three ties.  P-MADDPG beats QOLU+BUIA
        # (discordant, -1) but also beats LToS and MitM (concordant, +2).
        assert kendall_tau_on_values(values) == pytest.approx(1.0 / 6.0)

    def test_all_values_tied_scores_zero(self) -> None:
        values = {name: 0.4 for name in PUBLISHED_SCENARIO1_ORDER}
        assert kendall_tau_on_values(values) == pytest.approx(0.0)

    def test_the_published_order_scores_one(self) -> None:
        values = {"qolu_buia": 0.9, "pmaddpg": 0.7, "ltos": 0.5, "mitm": 0.3}
        assert kendall_tau_on_values(values) == pytest.approx(1.0)

    def test_the_reverse_order_scores_minus_one(self) -> None:
        values = {"qolu_buia": 0.3, "pmaddpg": 0.5, "ltos": 0.7, "mitm": 0.9}
        assert kendall_tau_on_values(values) == pytest.approx(-1.0)

    def test_methods_outside_the_published_four_are_ignored(self) -> None:
        values = {"qolu_buia": 0.9, "pmaddpg": 0.7, "ltos": 0.5, "mitm": 0.3}
        with_extra = {**values, "fixed_ltos": 1.0}
        assert kendall_tau_on_values(with_extra) == pytest.approx(
            kendall_tau_on_values(values)
        )

    def test_fewer_than_two_methods_has_no_pairs(self) -> None:
        assert kendall_tau_on_values({"qolu_buia": 0.5}) == pytest.approx(0.0)

    def test_a_single_item_has_no_pairs(self) -> None:
        assert kendall_tau(("a",), ("a",)) == pytest.approx(0.0)


class TestWindowScore:
    def test_published_ordering_scores_the_maximum(self) -> None:
        score = score_window(_PERFECT)
        assert score.first_place_count == 3
        assert score.kendall_sum == pytest.approx(3.0)
        assert score.reference_minmax_mean == pytest.approx(1.0)

    def test_reversed_ordering_scores_the_minimum(self) -> None:
        score = score_window(_REVERSED)
        assert score.first_place_count == 0
        assert score.kendall_sum == pytest.approx(-3.0)
        assert score.reference_minmax_mean == pytest.approx(0.0)

    def test_counts_only_metrics_where_the_reference_leads(self) -> None:
        values = dict(_PERFECT)
        values["urban"] = {
            "qolu_buia": 0.1,
            "pmaddpg": 0.5,
            "ltos": 0.3,
            "mitm": 0.7,
        }
        score = score_window(values)
        assert score.first_place_count == 2

    def test_a_flat_metric_gives_the_reference_a_half(self) -> None:
        values = {
            "land_use": {name: 0.4 for name in PUBLISHED_SCENARIO1_ORDER},
            "woodland": {name: 0.4 for name in PUBLISHED_SCENARIO1_ORDER},
            "urban": {name: 0.4 for name in PUBLISHED_SCENARIO1_ORDER},
        }
        score = score_window(values)
        assert score.reference_minmax_mean == pytest.approx(0.5)

    def test_sort_key_orders_primary_then_secondary_then_tertiary(self) -> None:
        strong = score_window(_PERFECT)
        weak = score_window(_REVERSED)
        assert strong.sort_key > weak.sort_key
        ranked = sorted([weak, strong], key=lambda item: item.sort_key, reverse=True)
        assert ranked[0] is strong

    def test_ties_on_the_primary_are_broken_by_kendall_tau(self) -> None:
        # Both have QOLU+BUIA first everywhere, but one keeps the published
        # baseline order and the other reverses it.
        tidy = _PERFECT
        messy = {
            metric: {
                "qolu_buia": values["qolu_buia"],
                "pmaddpg": 0.1,
                "ltos": 0.2,
                "mitm": 0.3,
            }
            for metric, values in _PERFECT.items()
        }
        assert score_window(tidy).first_place_count == 3
        assert score_window(messy).first_place_count == 3
        assert score_window(tidy).sort_key > score_window(messy).sort_key

    def test_record_is_json_serialisable(self) -> None:
        import json

        record = score_window(_PERFECT).as_record()
        assert json.loads(json.dumps(record)) == record

    def test_missing_methods_are_reported_rather_than_crashing(self) -> None:
        values = {"land_use": {"qolu_buia": 0.5, "pmaddpg": 0.2}}
        score = score_window(values)
        assert score.first_place_count == 1
        assert score.metrics[0].ranking == ("qolu_buia", "pmaddpg")


_THREE_WAY_TIE = {
    metric: {"qolu_buia": 0.2, "ltos": 0.2, "mitm": 0.2, "pmaddpg": 0.5}
    for metric in ("land_use", "woodland", "urban")
}


class TestPartialTies:
    """QOLU + BUIA tying with two baselines is the common case here."""

    def test_a_tie_is_not_a_first_place(self) -> None:
        tied_at_the_top = {
            metric: {"qolu_buia": 0.5, "ltos": 0.5, "mitm": 0.2, "pmaddpg": 0.1}
            for metric in ("land_use", "woodland", "urban")
        }
        score = score_window(tied_at_the_top)
        assert score.first_place_count == 0
        assert score.joint_first_count == 3

    def test_a_strict_win_counts_as_first_and_as_joint_first(self) -> None:
        score = score_window(_PERFECT)
        assert score.first_place_count == 3
        assert score.joint_first_count == 3

    def test_a_three_way_tie_below_pmaddpg_is_scored_on_the_pairs(self) -> None:
        score = score_window(_THREE_WAY_TIE)
        assert score.first_place_count == 0
        assert score.joint_first_count == 0
        assert score.kendall_sum == pytest.approx(3.0 / 6.0)

    def test_groups_tied_methods_together_best_group_first(self) -> None:
        score = score_window(_THREE_WAY_TIE)
        assert score.metrics[0].groups == (
            ("pmaddpg",),
            ("ltos", "mitm", "qolu_buia"),
        )

    def test_groups_of_a_strict_order_are_all_singletons(self) -> None:
        score = score_window(_PERFECT)
        assert score.metrics[0].groups == (
            ("qolu_buia",),
            ("pmaddpg",),
            ("ltos",),
            ("mitm",),
        )

    def test_groups_reach_the_record(self) -> None:
        record = score_window(_THREE_WAY_TIE).metrics[0].as_record()
        assert record["groups"] == [["pmaddpg"], ["ltos", "mitm", "qolu_buia"]]

    def test_the_name_tie_break_never_reaches_the_score(self) -> None:
        # "ltos" sorts before "qolu_buia", so a ranking-based score would hand
        # every tie to LToS.  Which of the tied methods is the reference must
        # not move any part of the score.
        as_qolu = score_window(_THREE_WAY_TIE, reference="qolu_buia")
        as_ltos = score_window(_THREE_WAY_TIE, reference="ltos")
        assert as_qolu.first_place_count == as_ltos.first_place_count == 0
        assert as_qolu.joint_first_count == as_ltos.joint_first_count == 0
        assert as_qolu.kendall_sum == pytest.approx(as_ltos.kendall_sum)
        assert as_qolu.reference_minmax_mean == pytest.approx(
            as_ltos.reference_minmax_mean
        )


class TestFlatMetricCells:
    """A metric on which every method scored the same carries no ordering."""

    def test_a_flat_cell_puts_nobody_first(self) -> None:
        score = score_window(_FLAT)
        assert score.first_place_count == 0
        assert all(not item.reference_first for item in score.metrics)

    def test_a_flat_cell_scores_zero_kendall_tau(self) -> None:
        score = score_window(_FLAT)
        assert score.kendall_sum == pytest.approx(0.0)

    def test_flat_cells_are_flagged_and_counted(self) -> None:
        score = score_window(_FLAT)
        assert score.flat_metrics == 3
        assert all(item.flat for item in score.metrics)
        assert all(item.as_record()["flat"] for item in score.metrics)

    def test_a_flat_cell_does_not_favour_an_alphabetically_early_method(
        self,
    ) -> None:
        # "ltos" sorts first and "qolu_buia" last, so a tie broken by name would
        # silently hand LToS the win on every flat metric.
        score = score_window(_FLAT, reference="ltos")
        assert score.first_place_count == 0
        assert score.kendall_sum == pytest.approx(0.0)

    def test_one_flat_metric_among_three_is_still_scored_on_the_others(
        self,
    ) -> None:
        values = dict(_PERFECT)
        values["woodland"] = {name: 0.0 for name in PUBLISHED_SCENARIO1_ORDER}
        score = score_window(values)
        assert score.first_place_count == 2
        assert score.flat_metrics == 1
        assert score.kendall_sum == pytest.approx(2.0)


class TestPatternMatchCount:
    def _summary(self, rankings: dict[int, dict[str, list[str]]]) -> dict:
        return {
            "reference_method": "qolu_buia",
            "configurations": [
                {
                    "label": "50x50_f1.0_cap10",
                    "scenarios": [
                        {"scenario": scenario, "ranking": ranking}
                        for scenario, ranking in sorted(rankings.items())
                    ],
                }
            ],
        }

    def test_counts_the_twelve_published_cells(self) -> None:
        published = {
            metric: list(PUBLISHED_SCENARIO1_ORDER)
            for metric in ("land_use", "woodland", "urban")
        }
        summary = self._summary({k: published for k in (1, 2, 3, 4)})
        matched, total, cells = count_pattern_matches(summary)
        assert (matched, total) == (12, 12)
        assert len(cells) == 12

    def test_counts_a_partial_match(self) -> None:
        published = {
            metric: list(PUBLISHED_SCENARIO1_ORDER)
            for metric in ("land_use", "woodland", "urban")
        }
        broken = dict(published)
        broken["urban"] = ["mitm", "ltos", "pmaddpg", "qolu_buia"]
        summary = self._summary({1: published, 2: published, 3: published, 4: broken})
        matched, total, _ = count_pattern_matches(summary)
        assert (matched, total) == (11, 12)
