"""Scoring a sensitivity cell, and the two ways of counting a 12-cell match."""

from __future__ import annotations

from typing import Any

import pytest

from chapter4_qolu.sensitivity_plan import SensitivityConfiguration
from chapter4_qolu.sensitivity_scoring import (
    ConfigurationScore,
    best_per_configuration,
    count_pattern_matches_tie_aware,
    score_configuration,
)
from chapter4_qolu.window_scan import ScanWindow

WINDOW = ScanWindow(easting_km=495, northing_km=140)
CONFIG = SensitivityConfiguration(
    environment="E495_N140", cap=1, reward_mode="per_parcel"
)


def _record(method: str, raw: dict[str, float], absolute: dict[str, float]) -> dict:
    return {
        "method": method,
        "final_metrics": dict(raw),
        "final_metrics_absolute": dict(absolute),
        "total_conversions": 100,
        "saturation_iteration": None,
        "runtime_seconds": 1.0,
    }


def _published_order_records() -> list[dict[str, Any]]:
    """Records whose RAW values put the methods in exactly the published order."""
    order = {"qolu_buia": 0.9, "pmaddpg": 0.8, "ltos": 0.7, "mitm": 0.6}
    return [
        _record(
            method,
            {"land_use": value, "woodland": value, "urban": value},
            {"land_use": value, "woodland": value, "urban": value},
        )
        for method, value in order.items()
    ]


class TestScoreConfiguration:
    def test_the_published_order_scores_three_first_places(self) -> None:
        score = score_configuration(_published_order_records(), CONFIG, WINDOW)
        assert score.window_score.first_place_count == 3
        assert score.window_score.kendall_sum == pytest.approx(3.0)

    def test_a_variant_is_scored_under_its_published_parent(self) -> None:
        records = _published_order_records()
        for record in records:
            if record["method"] == "ltos":
                record["method"] = "fixed_ltos"
            if record["method"] == "mitm":
                record["method"] = "mitm_learned"
        configuration = SensitivityConfiguration(
            environment="E495_N140",
            cap=1,
            reward_mode="per_parcel",
            ltos="fixed_ltos",
            mitm="mitm_learned",
        )
        score = score_configuration(records, configuration, WINDOW)
        assert score.window_score.first_place_count == 3
        assert score.n_records == 4

    def test_only_the_configurations_own_variants_are_used(self) -> None:
        records = _published_order_records()
        # A fixed-LToS record that would beat everything must be ignored by a
        # configuration that selected plain LToS.
        records.append(
            _record(
                "fixed_ltos",
                {"land_use": 9.0, "woodland": 9.0, "urban": 9.0},
                {"land_use": 9.0, "woodland": 9.0, "urban": 9.0},
            )
        )
        score = score_configuration(records, CONFIG, WINDOW)
        assert score.window_score.first_place_count == 3
        assert score.n_records == 4

    def test_value_closeness_uses_the_absolute_view(self) -> None:
        records = [
            _record(
                method,
                {"land_use": 0.0, "woodland": 0.0, "urban": 0.0},
                {"land_use": mean, "woodland": mean, "urban": mean},
            )
            for method, mean in (
                ("qolu_buia", 0.80),
                ("pmaddpg", 0.75),
                ("ltos", 0.70),
                ("mitm", 0.60),
            )
        ]
        score = score_configuration(records, CONFIG, WINDOW)
        # Published Scenario-1 Land Use column is exactly 0.80/0.75/0.70/0.60;
        # the other two columns are 0.05 and 0.10 higher.
        assert score.value_mae == pytest.approx((0.0 + 0.05 + 0.10) / 3.0)

    def test_an_incomplete_cell_scores_nothing(self) -> None:
        score = score_configuration([], CONFIG, WINDOW)
        assert score.n_records == 0
        assert score.value_mae is None
        assert score.window_score.first_place_count == 0

    def test_the_sort_key_prefers_first_places_then_tau_then_closeness(self) -> None:
        strong = ConfigurationScore.from_parts(
            CONFIG, WINDOW, first=2, tau=0.5, minmax=0.5, mae=0.9, mae_minmax=0.5, n=4
        )
        weak = ConfigurationScore.from_parts(
            CONFIG, WINDOW, first=1, tau=3.0, minmax=1.0, mae=0.0, mae_minmax=0.0, n=4
        )
        assert strong.sort_key > weak.sort_key
        closer = ConfigurationScore.from_parts(
            CONFIG, WINDOW, first=2, tau=0.5, minmax=0.5, mae=0.1, mae_minmax=0.5, n=4
        )
        assert closer.sort_key > strong.sort_key

    def test_a_missing_closeness_never_outranks_a_measured_one(self) -> None:
        measured = ConfigurationScore.from_parts(
            CONFIG, WINDOW, first=0, tau=0.0, minmax=0.0, mae=0.5, mae_minmax=0.5, n=4
        )
        missing = ConfigurationScore.from_parts(
            CONFIG, WINDOW, first=0, tau=0.0, minmax=0.0, mae=None, mae_minmax=None, n=0
        )
        assert measured.sort_key > missing.sort_key


class TestBestPerConfiguration:
    def test_each_configuration_keeps_its_best_window(self) -> None:
        other = ScanWindow(easting_km=495, northing_km=165)
        scores = [
            ConfigurationScore.from_parts(
                CONFIG,
                WINDOW,
                first=1,
                tau=0.0,
                minmax=0.0,
                mae=0.5,
                mae_minmax=0.5,
                n=4,
            ),
            ConfigurationScore.from_parts(
                CONFIG,
                other,
                first=3,
                tau=3.0,
                minmax=1.0,
                mae=0.1,
                mae_minmax=0.1,
                n=4,
            ),
        ]
        best = best_per_configuration(scores)
        assert len(best) == 1
        assert best[0].window.label == other.label

    def test_configurations_come_back_best_first(self) -> None:
        second = SensitivityConfiguration(
            environment="SE_region", cap=10, reward_mode="team"
        )
        scores = [
            ConfigurationScore.from_parts(
                CONFIG,
                WINDOW,
                first=1,
                tau=0.0,
                minmax=0.0,
                mae=0.5,
                mae_minmax=0.5,
                n=4,
            ),
            ConfigurationScore.from_parts(
                second,
                WINDOW,
                first=2,
                tau=0.0,
                minmax=0.0,
                mae=0.5,
                mae_minmax=0.5,
                n=4,
            ),
        ]
        best = best_per_configuration(scores)
        assert [item.configuration.label for item in best] == [
            second.label,
            CONFIG.label,
        ]


def _summary(values: dict[int, dict[str, dict[str, float]]]) -> dict[str, Any]:
    """A minimal campaign summary carrying RAW means and alphabetical rankings."""
    from chapter4_qolu.reporting.aggregate import rank_methods

    scenarios = []
    for scenario, per_metric in sorted(values.items()):
        methods = sorted({m for cell in per_metric.values() for m in cell})
        scenarios.append(
            {
                "scenario": scenario,
                "methods": {
                    method: {
                        "raw": {
                            metric: {"mean": per_metric[metric][method]}
                            for metric in per_metric
                            if method in per_metric[metric]
                        }
                    }
                    for method in methods
                },
                "ranking": {
                    metric: list(rank_methods(cell))
                    for metric, cell in per_metric.items()
                },
            }
        )
    return {
        "reference_method": "qolu_buia",
        "configurations": [{"label": "50x50_f1.0_cap1", "scenarios": scenarios}],
    }


class TestTieAwareMatching:
    def test_a_strict_published_order_matches_under_both_readings(self) -> None:
        cell = {"qolu_buia": 0.9, "pmaddpg": 0.8, "ltos": 0.7, "mitm": 0.6}
        values = {
            scenario: {
                metric: dict(cell) for metric in ("land_use", "woodland", "urban")
            }
            for scenario in (1, 2, 3, 4)
        }
        matched, total, _ = count_pattern_matches_tie_aware(_summary(values))
        assert (matched, total) == (12, 12)

    def test_an_all_tied_cell_is_not_a_strict_first_place(self) -> None:
        cell = {"qolu_buia": 0.5, "pmaddpg": 0.5, "ltos": 0.5, "mitm": 0.5}
        values = {1: {m: dict(cell) for m in ("land_use", "woodland", "urban")}}
        matched, total, cells = count_pattern_matches_tie_aware(_summary(values))
        # Scenario 1 emphasises all three metrics, so a tie is not a win.
        assert (matched, total) == (0, 3)
        assert all(cell["tied_top"] for cell in cells)

    def test_a_tie_away_from_the_emphasised_metric_still_counts(self) -> None:
        tied = {"qolu_buia": 0.8, "pmaddpg": 0.8, "ltos": 0.7, "mitm": 0.6}
        strict = {"qolu_buia": 0.9, "pmaddpg": 0.8, "ltos": 0.7, "mitm": 0.6}
        values = {
            2: {"land_use": strict, "woodland": tied, "urban": tied},
        }
        matched, total, _ = count_pattern_matches_tie_aware(_summary(values))
        assert (matched, total) == (3, 3)

    def test_the_alphabetical_count_is_the_campaign_one(self) -> None:
        from chapter4_qolu.scan_scoring import count_pattern_matches

        cell = {"qolu_buia": 0.5, "pmaddpg": 0.5, "ltos": 0.5, "mitm": 0.5}
        values = {1: {m: dict(cell) for m in ("land_use", "woodland", "urban")}}
        summary = _summary(values)
        alphabetical, _, _ = count_pattern_matches(summary)
        tie_aware, _, _ = count_pattern_matches_tie_aware(summary)
        # rank_methods breaks the tie by name, which puts ltos first and
        # qolu_buia last, so the alphabetical reading scores this cell 0 too -
        # but for the wrong reason, which is exactly why both are reported.
        assert alphabetical == 0
        assert tie_aware == 0
