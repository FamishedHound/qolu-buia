"""``run_scenarios --window``: launching a single-window campaign directly."""

from __future__ import annotations

from pathlib import Path

import pytest

from chapter4_qolu.campaign_plan import campaign_runs
from chapter4_qolu.run_scenarios import _build_parser, _settings_from
from chapter4_qolu.window_scan import ScanWindow, parse_window_label


class TestParseWindowLabel:
    def test_a_scan_label_round_trips(self) -> None:
        window = parse_window_label("E495_N140")
        assert window == ScanWindow(easting_km=495, northing_km=140)
        assert window.label == "E495_N140"

    def test_the_size_can_be_given_explicitly(self) -> None:
        window = parse_window_label("E495_N140", size_km=30)
        assert window.size_km == 30

    def test_a_lowercase_label_is_accepted(self) -> None:
        assert parse_window_label("e495_n140").label == "E495_N140"

    @pytest.mark.parametrize("text", ["E495", "N140_E495", "E495_N", "", "495_140"])
    def test_a_malformed_label_is_rejected(self, text: str) -> None:
        with pytest.raises(ValueError, match="window label"):
            parse_window_label(text)


class TestWindowOption:
    def _settings(self, tmp_path: Path, argv: list[str]):
        arguments = _build_parser().parse_args(argv)
        return _settings_from(arguments, tmp_path / "raster.tif")

    def test_the_window_reaches_every_run_spec(self, tmp_path: Path) -> None:
        settings = self._settings(
            tmp_path,
            [
                "--tag",
                "demo",
                "--window",
                "E495_N140",
                "--scenarios",
                "1",
                "--seeds",
                "0",
                "--configs",
                "50:1.0:10",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert settings.window_label == "E495_N140"
        # The loader is pinned to the *north-west* corner.
        assert settings.window_bng_km == (495.0, 190.0)
        for spec in campaign_runs(settings):
            assert spec.window_label == "E495_N140"
            assert spec.window_bng_km == (495.0, 190.0)

    def test_the_window_is_recorded_in_the_campaign_metadata(
        self, tmp_path: Path
    ) -> None:
        settings = self._settings(
            tmp_path,
            ["--window", "E470_N190", "--output-dir", str(tmp_path)],
        )
        metadata = settings.metadata()
        assert metadata["window_label"] == "E470_N190"
        assert metadata["window_bng_top_left_km"] == [470.0, 240.0]

    def test_without_the_option_nothing_is_pinned(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path, ["--output-dir", str(tmp_path)])
        assert settings.window_label is None
        assert settings.window_bng_km is None

    def test_the_window_size_follows_the_grid_size(self, tmp_path: Path) -> None:
        settings = self._settings(
            tmp_path,
            [
                "--window",
                "E495_N140",
                "--configs",
                "50:1.0:10",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert settings.window_bng_km == (495.0, 190.0)
