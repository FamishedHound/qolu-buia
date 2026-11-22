"""A run record has to say which learner produced it, not only which method.

``QOLUConfig.target_sync_interval`` (Algorithm 1's ``t_n``) is a choice of the
implementation that a run can override, so a record that names only the
method cannot tell a reader which ``t_n`` it ran under.  Every record therefore
carries the resolved :class:`~chapter4_qolu.config.QOLUConfig`, the learner
bookkeeping (gradient steps, how many times ``theta^- <- theta`` fired) and the
investor's :class:`~chapter4_qolu.config.BUIAConfig`, each ``None`` for a
method that has no such component.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.config import BUIAConfig, QOLUConfig
from chapter4_qolu.hparam_catalogue import resolve_hyperparameters
from chapter4_qolu.hparam_sensitivity import Perturbation
from chapter4_qolu.hparam_worker import HParamRunSpec, execute_hparam_run
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.scenario_worker import RunSpec, execute_run
from chapter4_qolu.tests.raster_fixtures import write_geotiff


@pytest.fixture(scope="module")
def raster(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(31337)
    return write_geotiff(
        tmp_path_factory.mktemp("provenance_raster")
        / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        rng.integers(1, 11, size=(16, 16), dtype=np.uint8),
        origin_e_m=400_000.0,
        origin_n_m=124_000.0,
    )


def _fields(raster: Path, method: str, output: Path) -> dict[str, object]:
    return {
        "raster": raster,
        "method": method,
        "scenario": 1,
        "seed": 0,
        "scale": ScaleConfig(grid_size=10, conversion_fraction=1.0, max_conversions=1),
        "output_path": output,
        "iterations": 8,
        "stakeholders": 2,
        "qolu_reward_mode": "team",
        "window_origin": (0, 0),
    }


@pytest.fixture(scope="module")
def records(raster: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    directory = tmp_path_factory.mktemp("provenance_runs")
    out = {}
    for method in ("qolu_buia", "mitm", "no_regulator"):
        execute_run(RunSpec(**_fields(raster, method, directory / f"{method}.json")))  # type: ignore[arg-type]
        # Read back off disk: what matters is what the file says.
        out[method] = json.loads(
            (directory / f"{method}.json").read_text(encoding="utf-8")
        )
    return out


class TestAQOLURecordNamesItsLearner:
    def test_it_carries_every_qolu_config_field(self, records: dict) -> None:
        stored = records["qolu_buia"]["qolu_config"]
        assert set(stored) == {item.name for item in fields(QOLUConfig)}

    def test_it_states_t_n(self, records: dict) -> None:
        stored = records["qolu_buia"]["qolu_config"]
        assert stored["target_sync_interval"] == QOLUConfig.target_sync_interval == 25

    def test_the_stored_config_rebuilds_the_one_that_ran(self, records: dict) -> None:
        stored = dict(records["qolu_buia"]["qolu_config"])
        stored["conv_channels"] = tuple(stored["conv_channels"])
        config = QOLUConfig(**stored)
        assert config.noisy_nets is True
        assert config.prioritised_replay is False
        assert (
            json.loads(json.dumps(asdict(config)))
            == records["qolu_buia"]["qolu_config"]
        )

    def test_it_counts_what_the_learner_did(self, records: dict) -> None:
        learner = records["qolu_buia"]["learner"]
        assert learner["target_sync_interval"] == 25
        assert len(learner["gradient_steps"]) == 2
        assert len(learner["target_syncs"]) == 2
        assert learner["total_target_syncs"] == sum(learner["target_syncs"])

    def test_it_carries_the_investor_config(self, records: dict) -> None:
        stored = records["qolu_buia"]["investor_config"]
        assert set(stored) == {item.name for item in fields(BUIAConfig)}
        assert stored["max_proposals"] == 1


class TestOtherMethodsSayTheyHaveNone:
    def test_a_baseline_without_qolu_or_investor(self, records: dict) -> None:
        record = records["mitm"]
        assert record["qolu_config"] is None
        assert record["learner"] is None
        assert record["investor_config"] is None

    def test_the_no_regulator_run_has_an_investor_and_no_qolu(
        self, records: dict
    ) -> None:
        record = records["no_regulator"]
        assert record["qolu_config"] is None
        assert record["learner"] is None
        assert record["investor_config"] == records["qolu_buia"]["investor_config"]


class TestAPerturbedRunRecordsThePerturbedValue:
    def test_the_hparam_path_stores_the_config_it_ran(
        self, raster: Path, tmp_path: Path
    ) -> None:
        item = resolve_hyperparameters(["target_sync_interval"])[0]
        perturbation = Perturbation(
            hyperparameter=item,
            direction=1,
            fraction=0.2,
            requested=30.0,
            value=30,
            rounded=False,
            clamped=False,
        )
        record = execute_hparam_run(
            HParamRunSpec(
                **_fields(raster, "qolu_buia", tmp_path / "perturbed.json"),  # type: ignore[arg-type]
                perturbation=perturbation,
            )
        )
        assert record["qolu_config"]["target_sync_interval"] == 30
        assert record["learner"]["target_sync_interval"] == 30
