"""The command line of the scenario x seed diagnostics sweep.

Split from ``test_diagnostics_sweep.py`` to keep both files inside the
package's line budget: the sweep's *behaviour* and the flags that ask for it
are separate things to break.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.run_distribution_diagnostics import _build_parser, main
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


class TestCommandLine:
    def test_the_parser_offers_the_sweep_flags(self) -> None:
        options = {
            action.option_strings[0]
            for action in _build_parser()._actions
            if action.option_strings
        }
        assert {"--scenarios", "--seeds", "--parallel", "--report-only"} <= options

    def test_the_cli_runs_a_sweep_when_given_plural_flags(
        self, tiny_class_raster: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "cli"
        assert (
            main(
                [
                    "--raster",
                    str(tiny_class_raster),
                    "--tag",
                    "cli",
                    "--window",
                    "E402_N110_6x6",
                    "--config",
                    "6:1.0:1",
                    "--scenarios",
                    "1",
                    "2",
                    "--seeds",
                    "0",
                    "--stakeholders",
                    "1",
                    "--iterations",
                    "2",
                    "--eval-iterations",
                    "2",
                    "--parallel",
                    "1",
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
        root = out / "distribution_diagnostics_cli"
        assert (root / "distribution_diagnostics_sweep.json").is_file()
        assert (root / "distribution_diagnostics_sweep.md").is_file()
        assert (root / "runs" / "distribution_diagnostics_cli_s2_seed0").is_dir()
