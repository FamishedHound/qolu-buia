"""End-to-end baseline runs on the UKCEH raster, plus interface conformance.

Every method is driven through :func:`chapter4_qolu.baselines.base.run_baseline`
inside the *same* environment object, under the same scale settings and from the
same seed, and is scored with the same metrics from
:mod:`chapter4_qolu.env.evaluation`.  That is the whole point of the baseline
package, so it is asserted directly here.

The raster is the Northern-Ireland placeholder product; the loader's
placeholder warning is expected and suppressed by the fixture.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.baselines import BASELINE_METHODS
from chapter4_qolu.baselines.base import (
    BaselineDecision,
    BaselineMethod,
    BaselineReport,
    run_baseline,
)
from chapter4_qolu.baselines.ltos import LToSMethod, fixed_ltos
from chapter4_qolu.baselines.mitm import MitMMethod, learned_mitm
from chapter4_qolu.baselines.pmaddpg import PMADDPGMethod
from chapter4_qolu.config import SCENARIOS, EnvConfig
from chapter4_qolu.data.lcm_loader import load_lcm_raster, select_region_of_interest
from chapter4_qolu.env.urban_env import APPROVE, REJECT, UrbanLandUseEnv

#: The integration crop: 30 x 30 parcels, small enough that a 100-iteration
#: run of three baselines stays inside a normal test budget.
GRID_SIZE = 30
ITERATIONS = 40

METRIC_KEYS = {
    "land_use",
    "woodland",
    "urban",
    "conversion_alignment",
    "c_align",
    "well_being",
}


def build_env(lcm_tif_path: Path, **overrides: object) -> UrbanLandUseEnv:
    """The shared environment: same grid, same scale settings for every method."""
    raster = load_lcm_raster(lcm_tif_path, warn_on_placeholder=False)
    grid = select_region_of_interest(raster, grid_size=GRID_SIZE)
    settings: dict[str, object] = {
        "grid_size": GRID_SIZE,
        "samples_per_iteration": 50,
        "max_conversions_per_iteration": 10,
        "iterations": ITERATIONS,
        "observation_window": 9,
    }
    settings.update(overrides)
    return UrbanLandUseEnv(
        grid, config=EnvConfig(**settings), weights=SCENARIOS[1]  # type: ignore[arg-type]
    )


def mitm_factory(env: UrbanLandUseEnv) -> MitMMethod:
    return MitMMethod.for_env(env, seed=0)


def ltos_factory(env: UrbanLandUseEnv) -> LToSMethod:
    return LToSMethod.for_env(
        env, seed=0, districts_per_side=6, batch_size=8, learning_starts=8
    )


def pmaddpg_factory(env: UrbanLandUseEnv) -> PMADDPGMethod:
    return PMADDPGMethod.for_env(
        env, seed=0, batch_size=64, learning_starts=64, replay_capacity=20_000
    )


def fixed_ltos_factory(env: UrbanLandUseEnv) -> LToSMethod:
    return fixed_ltos(
        env, seed=0, districts_per_side=6, batch_size=8, learning_starts=8
    )


def learned_mitm_factory(env: UrbanLandUseEnv) -> MitMMethod:
    return learned_mitm(env, seed=0)


HEADLINE_FACTORIES: dict[str, Callable[[UrbanLandUseEnv], BaselineMethod]] = {
    "mitm": mitm_factory,
    "ltos": ltos_factory,
    "p_maddpg": pmaddpg_factory,
}
ALL_FACTORIES: dict[str, Callable[[UrbanLandUseEnv], BaselineMethod]] = {
    **HEADLINE_FACTORIES,
    "fixed_ltos": fixed_ltos_factory,
    "mitm_learned": learned_mitm_factory,
}


class TestInterfaceConformance:
    """Every method has to be drivable by the same code."""

    def test_the_package_exposes_the_paper_s_three_baselines(self) -> None:
        assert BASELINE_METHODS == (MitMMethod, LToSMethod, PMADDPGMethod)

    @pytest.mark.parametrize("name", sorted(ALL_FACTORIES))
    def test_every_method_implements_the_interface(
        self, name: str, lcm_tif_path: Path
    ) -> None:
        env = build_env(lcm_tif_path, grid_size=GRID_SIZE)
        method = ALL_FACTORIES[name](env)
        assert isinstance(method, BaselineMethod)
        assert method.name == name
        batch = env.reset(seed=0)
        method.reset(env, 0)
        decision = method.decide(batch)
        assert isinstance(decision, BaselineDecision)
        assert decision.decisions.shape == (len(batch.parcels),)
        assert set(np.unique(decision.decisions)) <= {REJECT, APPROVE}
        if decision.proposals is not None:
            assert set(decision.proposals) <= set(batch.parcels)
        result = env.step(decision.decisions, proposals=decision.proposals)
        method.observe(result)
        loss = method.train_step()
        assert loss is None or np.isfinite(loss)

    @pytest.mark.parametrize("name", sorted(ALL_FACTORIES))
    def test_every_method_reads_the_shared_conversion_budget(
        self, name: str, lcm_tif_path: Path
    ) -> None:
        env = build_env(lcm_tif_path, max_conversions_per_iteration=3, iterations=5)
        report = run_baseline(env, ALL_FACTORIES[name](env), iterations=5, seed=0)
        assert max(report.n_conversions) <= 3


@pytest.mark.parametrize("name", sorted(HEADLINE_FACTORIES))
def test_baseline_runs_end_to_end_on_the_raster(name: str, lcm_tif_path: Path) -> None:
    """One integration run per baseline, same env settings and seed for all."""
    env = build_env(lcm_tif_path)
    report = run_baseline(env, HEADLINE_FACTORIES[name](env), ITERATIONS, seed=0)

    assert isinstance(report, BaselineReport)
    assert report.method == name
    assert report.iterations == ITERATIONS
    assert len(report.team_rewards) == ITERATIONS
    assert all(np.isfinite(value) for value in report.team_rewards)
    assert all(np.isfinite(value) for value in report.losses)
    assert set(report.final_metrics) == METRIC_KEYS
    assert all(np.isfinite(value) for value in report.final_metrics.values())
    assert 0.0 <= report.final_metrics["woodland"] <= 1.0
    assert report.total_conversions > 0
    assert report.urban_fraction[-1] > report.urban_fraction[0]


class TestComparability:
    def test_all_methods_see_the_same_first_decision_buffer(
        self, lcm_tif_path: Path
    ) -> None:
        buffers = []
        for factory in HEADLINE_FACTORIES.values():
            env = build_env(lcm_tif_path)
            method = factory(env)
            batch = env.reset(seed=11)
            method.reset(env, 11)
            method.decide(batch)
            buffers.append(batch.parcels)
        assert all(parcels == buffers[0] for parcels in buffers)

    def test_all_methods_report_the_same_metric_columns(
        self, lcm_tif_path: Path
    ) -> None:
        columns = []
        for factory in HEADLINE_FACTORIES.values():
            env = build_env(lcm_tif_path, iterations=4)
            report = run_baseline(env, factory(env), iterations=4, seed=12)
            columns.append(set(report.final_metrics))
        assert all(column == METRIC_KEYS for column in columns)

    def test_the_methods_do_not_all_behave_identically(
        self, lcm_tif_path: Path
    ) -> None:
        rewards = []
        for factory in HEADLINE_FACTORIES.values():
            env = build_env(lcm_tif_path, iterations=8)
            rewards.append(run_baseline(env, factory(env), 8, seed=13).team_rewards)
        assert not all(np.allclose(series, rewards[0]) for series in rewards[1:])
