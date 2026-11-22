"""The ``no_regulator`` ablation.

BUIA proposes and every proposal is converted: the top-down layer is removed
and nothing else.  The investor is the one QOLU + BUIA uses - same
:class:`~chapter4_qolu.config.BUIAConfig`, same seed, same reward
``R_NN = Attract(s') - Attract(s)``, same decision buffer - so a paired
comparison of the two isolates what the stakeholders' appraisal does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from chapter4_qolu.agent.buia_agent import BUIAAgent, buia_proposal_limit
from chapter4_qolu.agent.qolu_agent import QOLUAgent
from chapter4_qolu.baselines.base import run_baseline
from chapter4_qolu.baselines.no_regulator import NoRegulatorMethod
from chapter4_qolu.campaign_plan import METHOD_COST_SECONDS
from chapter4_qolu.config import BUIAConfig, QOLUConfig
from chapter4_qolu.methods import (
    METHOD_KEYS,
    NO_REGULATOR_METHOD,
    PUBLISHED_METHODS,
    canonical_method,
    is_variant,
)
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.reporting.tables import METHOD_LABELS, PAPER_ROW_ORDER
from chapter4_qolu.scenario_worker import RunSpec, execute_run
from chapter4_qolu.tests.baseline_fixtures import make_env
from chapter4_qolu.tests.raster_fixtures import write_geotiff


def investor_env():
    return make_env(
        samples_per_iteration=8,
        max_conversions_per_iteration=1,
        iterations=10,
    )


class TestTheRegistry:
    def test_it_is_a_method_key_of_its_own(self) -> None:
        assert NO_REGULATOR_METHOD == "no_regulator"
        assert NO_REGULATOR_METHOD in METHOD_KEYS
        assert canonical_method(NO_REGULATOR_METHOD) == NO_REGULATOR_METHOD
        assert not is_variant(NO_REGULATOR_METHOD)

    def test_it_is_not_a_published_method(self) -> None:
        assert NO_REGULATOR_METHOD not in PUBLISHED_METHODS

    def test_it_is_priced_for_the_plan(self) -> None:
        assert METHOD_COST_SECONDS[NO_REGULATOR_METHOD] > 0.0

    def test_it_is_the_first_row_of_every_table(self) -> None:
        assert PAPER_ROW_ORDER[0] == NO_REGULATOR_METHOD
        assert METHOD_LABELS[NO_REGULATOR_METHOD] == "No regulator"


class TestTheInvestorIsTheOneQOLUBUIAUses:
    def test_the_same_seed_builds_the_same_initial_network(self) -> None:
        env = investor_env()
        method = NoRegulatorMethod.for_env(env, seed=3)
        # QOLU + BUIA builds its stakeholders first; BUIA reseeds torch itself,
        # so its initial weights must not depend on what was built before it.
        QOLUAgent(
            QOLUConfig(
                observation_channels=env.observation_channels,
                observation_window=env.config.observation_window,
            ),
            seed=3,
        )
        reference = BUIAAgent(
            BUIAConfig(
                observation_channels=env.buia_observation_channels,
                observation_window=env.config.buia_observation_window,
                max_proposals=buia_proposal_limit(env.config),
            ),
            seed=3,
        )
        built = method.investor.network.state_dict()
        expected = reference.network.state_dict()
        assert all(torch.equal(built[name], expected[name]) for name in expected)
        assert method.investor.config == reference.config

    def test_it_learns_from_its_own_reward(self) -> None:
        env = investor_env()
        method = NoRegulatorMethod.for_env(env, seed=0)
        before = {
            name: tensor.clone()
            for name, tensor in method.investor.network.state_dict().items()
        }
        report = run_baseline(env, method, iterations=10, seed=0)
        after = method.investor.network.state_dict()
        assert method.investor.updates == 10
        assert len(report.losses) == 10
        assert not all(torch.equal(before[name], after[name]) for name in before)


class TestEveryProposalIsConverted:
    def test_each_iteration_converts_what_the_investor_proposed(self) -> None:
        env = investor_env()
        report = run_baseline(env, NoRegulatorMethod.for_env(env, seed=0), 10, seed=0)
        assert report.n_proposals == (1,) * 10
        assert report.n_conversions == report.n_proposals
        assert report.n_approved == report.n_proposals

    def test_nothing_is_appraised(self) -> None:
        """Approved exactly the proposals: no candidate is rejected on merit."""
        env = investor_env()
        method = NoRegulatorMethod.for_env(env, seed=0)
        batch = env.reset(seed=0)
        method.reset(env, 0)
        decision = method.decide(batch)
        proposed = set(decision.proposals)
        approved = {
            parcel
            for parcel, action in zip(batch.parcels, decision.decisions)
            if action == 1
        }
        assert approved == proposed
        assert len(proposed) == 1

    def test_without_a_cap_every_proposal_still_converts(self) -> None:
        env = make_env(
            samples_per_iteration=8, max_conversions_per_iteration=None, iterations=4
        )
        report = run_baseline(env, NoRegulatorMethod.for_env(env, seed=1), 4, seed=1)
        assert report.n_conversions == report.n_proposals
        assert sum(report.n_conversions) > 4

    def test_a_run_is_deterministic_given_its_seed(self) -> None:
        reports = []
        for _ in range(2):
            env = investor_env()
            reports.append(
                run_baseline(env, NoRegulatorMethod.for_env(env, seed=5), 10, seed=5)
            )
        assert reports[0].team_rewards == reports[1].team_rewards
        assert reports[0].final_metrics == reports[1].final_metrics


@pytest.fixture(scope="module")
def tiny_raster(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(20261003)
    return write_geotiff(
        tmp_path_factory.mktemp("no_regulator_raster")
        / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        rng.integers(1, 11, size=(16, 16), dtype=np.uint8),
        origin_e_m=400_000.0,
        origin_n_m=124_000.0,
    )


def _spec(raster: Path, method: str, output: Path) -> RunSpec:
    return RunSpec(
        raster=raster,
        method=method,
        scenario=2,
        seed=1,
        scale=ScaleConfig(grid_size=10, conversion_fraction=1.0, max_conversions=1),
        output_path=output,
        iterations=12,
        stakeholders=1,
        qolu_reward_mode="team",
        window_origin=(0, 0),
    )


class TestTheCampaignRunsIt:
    def test_a_campaign_run_writes_an_ordinary_record(
        self, tiny_raster: Path, tmp_path: Path
    ) -> None:
        record = execute_run(
            _spec(tiny_raster, NO_REGULATOR_METHOD, tmp_path / "no_regulator.json")
        )
        assert (tmp_path / "no_regulator.json").is_file()
        assert record["method"] == NO_REGULATOR_METHOD
        assert record["total_conversions"] == 12
        assert record["per_iteration"]["n_proposals"] == [1] * 12
        assert set(record["final_metrics"]) >= {"land_use", "woodland", "urban"}
        assert "c_align" in record["final_metrics"]

    def test_it_sees_the_same_landscape_as_qolu_buia(
        self, tiny_raster: Path, tmp_path: Path
    ) -> None:
        mine = execute_run(_spec(tiny_raster, NO_REGULATOR_METHOD, tmp_path / "a.json"))
        theirs = execute_run(_spec(tiny_raster, "qolu_buia", tmp_path / "b.json"))
        assert mine["initial_grid_digest"] == theirs["initial_grid_digest"]
        assert mine["initial_grid"] == theirs["initial_grid"]
