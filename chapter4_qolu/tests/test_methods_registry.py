"""Tests for the torch-free method registry and the P-MADDPG mask variant."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chapter4_qolu.methods import (
    METHOD_KEYS,
    PMADDPG_METHOD_VARIANTS,
    QOLU_METHOD,
    VARIANT_PARENTS,
    canonical_method,
)
from chapter4_qolu.reporting.paths import parse_scale_config
from chapter4_qolu.reporting.tables import METHOD_LABELS, PAPER_ROW_ORDER
from chapter4_qolu.tests.raster_fixtures import write_geotiff


class TestRegistry:
    def test_every_published_and_variant_method_is_registered(self) -> None:
        assert QOLU_METHOD in METHOD_KEYS
        for method in ("pmaddpg", "pmaddpg_v1", "pmaddpg_reg_only"):
            assert method in METHOD_KEYS
        for method in ("ltos", "fixed_ltos", "mitm", "mitm_learned"):
            assert method in METHOD_KEYS

    def test_variants_map_onto_their_published_parent(self) -> None:
        assert canonical_method("pmaddpg_v1") == "pmaddpg"
        assert canonical_method("pmaddpg_reg_only") == "pmaddpg"
        assert canonical_method("fixed_ltos") == "ltos"
        assert canonical_method("mitm_learned") == "mitm"

    def test_the_pmaddpg_keys_name_their_variant(self) -> None:
        assert PMADDPG_METHOD_VARIANTS["pmaddpg"] == "published"
        assert PMADDPG_METHOD_VARIANTS["pmaddpg_v1"] == "v1"
        assert PMADDPG_METHOD_VARIANTS["pmaddpg_reg_only"] == "reg_only"
        assert set(PMADDPG_METHOD_VARIANTS) <= set(METHOD_KEYS)

    def test_a_published_method_is_its_own_canonical_form(self) -> None:
        for method in ("qolu_buia", "pmaddpg", "ltos", "mitm"):
            assert canonical_method(method) == method

    def test_every_variant_parent_is_itself_a_method(self) -> None:
        for variant, parent in VARIANT_PARENTS.items():
            assert variant in METHOD_KEYS
            assert parent in METHOD_KEYS

    def test_scenario_worker_re_exports_the_registry(self) -> None:
        from chapter4_qolu.scenario_worker import METHOD_KEYS as worker_keys

        assert worker_keys == METHOD_KEYS

    def test_the_table_row_order_keeps_variants_next_to_their_parent(self) -> None:
        assert PAPER_ROW_ORDER.index("pmaddpg_v1") == (
            PAPER_ROW_ORDER.index("pmaddpg") + 1
        )
        assert PAPER_ROW_ORDER.index("pmaddpg_reg_only") == (
            PAPER_ROW_ORDER.index("pmaddpg") + 2
        )
        assert "P-MADDPG" in METHOD_LABELS["pmaddpg_v1"]
        assert "P-MADDPG" in METHOD_LABELS["pmaddpg_reg_only"]

    def test_the_cost_table_prices_every_variant(self) -> None:
        from chapter4_qolu.campaign_plan import METHOD_COST_SECONDS

        for method in METHOD_KEYS:
            assert method in METHOD_COST_SECONDS


@pytest.fixture()
def tiny_class_raster(tmp_path: Path) -> Path:
    rng = np.random.default_rng(20260924)
    codes = rng.integers(1, 11, size=(16, 16), dtype=np.uint8)
    return write_geotiff(
        tmp_path / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        codes,
        origin_e_m=400_000.0,
        origin_n_m=120_000.0,
    )


class TestPMADDPGVariantRuns:
    def test_the_default_variant_runs(
        self, tiny_class_raster: Path, tmp_path: Path
    ) -> None:
        from chapter4_qolu.scenario_worker import RunSpec, execute_run

        spec = RunSpec(
            raster=tiny_class_raster,
            method="pmaddpg",
            scenario=1,
            seed=0,
            scale=parse_scale_config("8:1.0:10"),
            output_path=tmp_path / "pmaddpg_seed0.json",
            iterations=2,
            stakeholders=1,
        )
        record = execute_run(spec)
        assert record["method"] == "pmaddpg"
        assert set(record["final_metrics"]) >= {"land_use", "woodland", "urban"}

    def test_a_non_default_variant_either_runs_or_says_why_not(
        self, tiny_class_raster: Path, tmp_path: Path
    ) -> None:
        from chapter4_qolu.scenario_worker import (
            RunSpec,
            _pmaddpg_variants_available,
            execute_run,
        )

        spec = RunSpec(
            raster=tiny_class_raster,
            method="pmaddpg_v1",
            scenario=1,
            seed=0,
            scale=parse_scale_config("8:1.0:10"),
            output_path=tmp_path / "pmaddpg_v1_seed0.json",
            iterations=2,
            stakeholders=1,
        )
        if _pmaddpg_variants_available():
            assert execute_run(spec)["method"] == "pmaddpg_v1"
        else:
            with pytest.raises(ValueError, match="revised P-MADDPG"):
                execute_run(spec)
