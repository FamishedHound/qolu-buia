"""Stage B2: the winning recipe on every window, and the separation statistic."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from chapter4_qolu.reporting.published import published_means
from chapter4_qolu.reporting.sensitivity_b2 import composition_note, stage_b2_section
from chapter4_qolu.run_record import encode_class_grid
from chapter4_qolu.sensitivity_b2 import (
    PUBLISHED_SEPARATION,
    SEPARATION_TOLERANCE,
    B2Score,
    score_b2_window,
    separation,
)
from chapter4_qolu.window_scan import ScanWindow

WINDOW = ScanWindow(easting_km=495, northing_km=165)


def _record(method: str, value: float) -> dict[str, Any]:
    rng = np.random.default_rng(abs(hash(method)) % 2**32)
    initial = rng.integers(0, 10, size=(8, 8)).astype(np.int64)
    final = initial.copy()
    final[0, :] = 9
    return {
        "method": method,
        "scenario": 1,
        "seed": 0,
        "final_metrics": {
            "land_use": value,
            "woodland": value,
            "urban": value,
        },
        "final_metrics_absolute": {
            "land_use": value,
            "woodland": value,
            "urban": value,
        },
        "initial_grid": encode_class_grid(initial),
        "final_grid": encode_class_grid(final),
    }


class TestSeparation:
    def test_the_published_separation_is_one_fifteenth(self) -> None:
        # Table 4's adjacent gaps are 0.05, 0.05 and 0.10 in all three columns.
        assert PUBLISHED_SEPARATION == pytest.approx(1.0 / 15.0)
        assert separation(published_means(1)) == pytest.approx(PUBLISHED_SEPARATION)

    def test_identical_methods_separate_by_nothing(self) -> None:
        values = {
            metric: {m: 0.5 for m in ("qolu_buia", "pmaddpg", "ltos", "mitm")}
            for metric in ("land_use", "woodland", "urban")
        }
        assert separation(values) == pytest.approx(0.0)

    def test_an_empty_cell_has_no_separation(self) -> None:
        assert separation({}) is None

    def test_the_statistic_is_the_mean_adjacent_gap(self) -> None:
        values = {"land_use": {"a": 1.0, "b": 0.8, "c": 0.5, "d": 0.1}}
        # gaps 0.2, 0.3, 0.4 -> mean 0.3
        assert separation(values) == pytest.approx(0.3)


def _record_without_grids(method: str, value: float) -> dict[str, Any]:
    """A record that cannot be re-scored, so the stored values are used."""
    record = _record(method, value)
    del record["initial_grid"]
    del record["final_grid"]
    return record


class TestB2Score:
    def _score(self, order: dict[str, float]) -> B2Score:
        records = [_record(method, value) for method, value in order.items()]
        return score_b2_window(records, WINDOW)

    def _score_stored(self, order: dict[str, float]) -> B2Score:
        records = [
            _record_without_grids(method, value) for method, value in order.items()
        ]
        return score_b2_window(records, WINDOW)

    def test_the_published_order_qualifies(self) -> None:
        score = self._score(
            {"qolu_buia": 0.9, "pmaddpg": 0.8, "ltos": 0.7, "mitm": 0.6}
        )
        assert score.qualifies is True
        assert score.window_score.first_place_count == 3

    def test_a_degenerate_window_does_not_qualify(self) -> None:
        score = self._score(
            {"qolu_buia": 0.5, "pmaddpg": 0.5, "ltos": 0.5, "mitm": 0.5}
        )
        assert score.qualifies is False

    def test_a_missing_cell_scores_nothing(self) -> None:
        score = score_b2_window([], WINDOW)
        assert score.n_records == 0
        assert score.value_mae is None
        assert score.qualifies is False

    def test_the_best_saturation_distance_is_recorded(self) -> None:
        score = self._score(
            {"qolu_buia": 0.9, "pmaddpg": 0.8, "ltos": 0.7, "mitm": 0.6}
        )
        assert score.max_distance_km in (5.0, 10.0, 20.0)
        assert score.value_mae is not None

    def test_a_qualifying_window_outranks_a_closer_but_unordered_one(self) -> None:
        ordered = B2Score.__new__(B2Score)
        good = self._score({"qolu_buia": 0.9, "pmaddpg": 0.8, "ltos": 0.7, "mitm": 0.6})
        bad = self._score({"mitm": 0.9, "ltos": 0.8, "pmaddpg": 0.7, "qolu_buia": 0.6})
        del ordered
        assert good.sort_key > bad.sort_key

    def test_the_separation_ratio_is_against_the_published_gap(self) -> None:
        # Records without grids cannot be re-scored, so the stored values are
        # what the separation is measured on.
        score = self._score_stored(
            {"qolu_buia": 0.9, "pmaddpg": 0.8, "ltos": 0.7, "mitm": 0.6}
        )
        # Adjacent gaps of 0.1 against a published mean of 1/15.
        assert score.separation_ratio == pytest.approx(0.1 / PUBLISHED_SEPARATION)
        assert score.separation_in_band is True

    def test_a_tiny_separation_falls_out_of_band(self) -> None:
        score = self._score_stored(
            {"qolu_buia": 0.503, "pmaddpg": 0.502, "ltos": 0.501, "mitm": 0.500}
        )
        assert score.separation_ratio < 1.0 / SEPARATION_TOLERANCE
        assert score.separation_in_band is False

    def test_stored_grids_are_preferred_to_stored_values(self) -> None:
        """B2 re-scores from the grids; the recorded values are the fallback."""
        order = {"qolu_buia": 0.9, "pmaddpg": 0.8, "ltos": 0.7, "mitm": 0.6}
        assert self._score(order).separation != pytest.approx(
            self._score_stored(order).separation
        )


class TestReport:
    def _entries(self) -> list[dict[str, Any]]:
        score = score_b2_window(
            [
                _record(m, v)
                for m, v in (
                    ("qolu_buia", 0.9),
                    ("pmaddpg", 0.8),
                    ("ltos", 0.7),
                    ("mitm", 0.6),
                )
            ],
            WINDOW,
        )
        return [
            {
                **score.as_record(),
                "tag": "b2_E495_N165",
                "campaign_dir": "x",
                "methods": {
                    "qolu_buia": {
                        "absolute": {
                            "land_use": 0.98,
                            "woodland": 0.33,
                            "urban": 0.843,
                        }
                    },
                    "mitm": {
                        "absolute": {
                            "land_use": 0.969,
                            "woodland": 0.259,
                            "urban": 0.838,
                        }
                    },
                },
            }
        ]

    def test_the_table_lists_every_window(self) -> None:
        text = "\n".join(
            stage_b2_section(
                {
                    "recipe": {"label": "demo", "reading": "plain"},
                    "fitted_window": "E495_N165",
                    "n_windows": 1,
                    "windows": self._entries(),
                }
            )
        )
        assert "E495_N165" in text
        assert "Sep/pub" in text
        assert "1 of 1 windows reproduce the published ordering" in text

    def test_no_records_is_said_plainly(self) -> None:
        text = "\n".join(stage_b2_section({"windows": []}))
        assert "No stage-B2 records were found." in text

    def test_the_composition_note_quotes_the_real_spreads(self) -> None:
        text = "\n".join(composition_note(self._entries(), "E495_N165"))
        assert "Why the numbers sit where they do" in text
        assert "land_use 0.969-0.980" in text
        assert "spread 0.011" in text

    def test_the_composition_note_needs_the_fitted_window(self) -> None:
        assert composition_note(self._entries(), "nowhere") == []
