"""The perturbed code path must differ from the baseline's *only* in the value.

An agent-side perturbation cannot go through
:func:`chapter4_qolu.scenario_worker.execute_run`, because that function builds
the :class:`~chapter4_qolu.config.QOLUConfig` internally from the method key
alone.  :func:`chapter4_qolu.hparam_worker._perturbed_qolu_run` therefore
repeats the body with the config perturbed, and its docstring promises the two
"must produce byte-comparable records".

Building the same objects in a *different order* would break that: both
``QOLUAgent`` and ``BUIAAgent`` draw their weight initialisation from the
process-wide torch generator, so the order decides every initial weight in the
run.  The whole analysis is a paired comparison of a perturbed run against a
baseline run, so a difference of that kind would be attributed to the
hyper-parameter and be indistinguishable from its effect.

The invariant below is the one that catches it: a perturbation whose value is
the package default must reproduce the baseline run **exactly**.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.hparam_catalogue import resolve_hyperparameters
from chapter4_qolu.hparam_sensitivity import Perturbation
from chapter4_qolu.hparam_worker import HParamRunSpec, execute_hparam_run
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.scenario_worker import RunSpec, execute_run
from chapter4_qolu.tests.raster_fixtures import write_geotiff

METRICS = ("land_use", "woodland", "urban")


@pytest.fixture(scope="module")
def parity_raster(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(4242)
    return write_geotiff(
        tmp_path_factory.mktemp("parity_raster")
        / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        rng.integers(1, 11, size=(24, 24), dtype=np.uint8),
        origin_e_m=400_000.0,
        origin_n_m=124_000.0,
    )


def _identity_perturbation(name: str) -> Perturbation:
    """A perturbation that configures the package default it perturbs.

    Its ``value`` *is* the baseline, so the run it produces has to match the
    baseline run; only the code path differs.
    """
    item = resolve_hyperparameters([name])[0]
    return Perturbation(
        hyperparameter=item,
        direction=1,
        fraction=0.2,
        requested=float(item.baseline) * 1.2,
        value=item.baseline,
        rounded=True,
        clamped=False,
    )


def _fields(raster: Path, output: Path) -> dict[str, object]:
    """One cell, big enough to *discriminate*.

    Design note (why not the 6 x 6 fixture the other hparam tests use): the
    divergence this pins is a difference in the initial network weights, which
    only shows up once the stakeholders disagree often enough for the consensus
    sign to flip on some parcel.  On a 6 x 6 grid with two stakeholders two
    differently-ordered code paths can agree anyway, so the test would pass
    with the divergence present.  Three stakeholders on a 20 x 20 grid
    separates them, and still runs in a few seconds.
    """
    return {
        "raster": raster,
        "method": "qolu_buia",
        "scenario": 1,
        "seed": 0,
        "scale": ScaleConfig(grid_size=20, conversion_fraction=1.0, max_conversions=2),
        "output_path": output,
        "iterations": 25,
        "stakeholders": 3,
        "qolu_reward_mode": "team",
        "window_origin": (0, 0),
    }


class TestTheTwoCodePathsAgree:
    @pytest.fixture(scope="class")
    def records(
        self, parity_raster: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> tuple[dict, dict]:
        directory = tmp_path_factory.mktemp("parity_runs")
        baseline = execute_run(
            RunSpec(**_fields(parity_raster, directory / "baseline.json"))  # type: ignore[arg-type]
        )
        perturbed = execute_hparam_run(
            HParamRunSpec(
                **_fields(parity_raster, directory / "identity.json"),  # type: ignore[arg-type]
                perturbation=_identity_perturbation("target_sync_interval"),
            )
        )
        return baseline, perturbed

    def test_the_identity_perturbation_took_the_perturbed_path(
        self, records: tuple[dict, dict]
    ) -> None:
        _, perturbed = records
        assert perturbed["qolu_overrides"] == {"target_sync_interval": 25}

    def test_both_runs_saw_the_same_landscape(self, records: tuple[dict, dict]) -> None:
        baseline, perturbed = records
        assert baseline["initial_grid_digest"] == perturbed["initial_grid_digest"]

    def test_the_final_metrics_are_identical(self, records: tuple[dict, dict]) -> None:
        baseline, perturbed = records
        assert [perturbed["final_metrics"][k] for k in METRICS] == [
            baseline["final_metrics"][k] for k in METRICS
        ]

    def test_the_whole_trajectory_is_identical(
        self, records: tuple[dict, dict]
    ) -> None:
        # Not just the endpoint: a divergence that cancelled out by the last
        # iteration would still be a divergence.
        baseline, perturbed = records
        assert perturbed["per_iteration"] == baseline["per_iteration"]

    def test_the_final_grid_is_identical(self, records: tuple[dict, dict]) -> None:
        baseline, perturbed = records
        assert perturbed["final_grid"] == baseline["final_grid"]
