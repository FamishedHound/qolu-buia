"""A perturbation that changed nothing must not be counted as evidence.

The analysis asks whether the method ranking survives a +/-20 % move.  A
perturbation whose runs come back **bit-identical** to the baseline's has not
tested that at all: its ranking is preserved by construction, not by
experiment.  That happens for real, at real settings - a replay buffer of
80 000 transitions never evicts anything in a run that stores 50 000, and a
target network synced every 1 200 gradient steps is never synced in a run that
takes 980 - so the payload and the report have to tell the two apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chapter4_qolu.hparam_plan import HParamSettings
from chapter4_qolu.hparam_summary import (
    INERT_REASON,
    inert_perturbation,
    informative_perturbations,
)
from chapter4_qolu.reporting.hparam_report import render_hparam_report
from chapter4_qolu.reporting.hparam_tables import overall_section, verdict_table
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.run_hparam_sensitivity import run_hparam_sensitivity
from chapter4_qolu.tests.raster_fixtures import write_geotiff


@pytest.fixture(scope="module")
def tiny_hparam_summary(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A three-iteration analysis, run once and read by every assertion below."""
    rng = np.random.default_rng(20260929)
    raster: Path = write_geotiff(
        tmp_path_factory.mktemp("inert_raster")
        / "lcm2015_gb_1km_dominant_aggregate_class.tif",
        rng.integers(1, 11, size=(24, 24), dtype=np.uint8),
        origin_e_m=400_000.0,
        origin_n_m=124_000.0,
    )
    return run_hparam_sensitivity(
        HParamSettings(
            raster=raster,
            tag="inert",
            hyperparameters=("gamma", "samples_per_iteration"),
            methods=("qolu_buia", "mitm"),
            scenarios=(1,),
            seeds=(0, 1, 2),
            iterations=3,
            stakeholders=1,
            scale=ScaleConfig(grid_size=6, conversion_fraction=1.0, max_conversions=2),
            window="E402_N110_6x6",
            parallel=1,
            output_dir=tmp_path_factory.mktemp("inert_results"),
        )
    )


@pytest.fixture(scope="module")
def tiny_hparam_report(tiny_hparam_summary: dict[str, Any]) -> str:
    return render_hparam_report(tiny_hparam_summary)


def _cell(
    values: list[float], baseline_values: list[float] | None, borrowed: bool = False
) -> dict[str, Any]:
    return {
        "methods": {
            "qolu_buia": {
                "values": values,
                "baseline_values": baseline_values,
                "borrowed_from_baseline": borrowed,
            }
        },
        "ranking": ["qolu_buia"],
        "baseline_ranking": ["qolu_buia"],
        "ranking_preserved": True,
        "kendall_tau": 1.0,
    }


def _scenarios(
    values: list[float], baseline_values: list[float] | None, borrowed: bool = False
) -> list[dict[str, Any]]:
    return [
        {
            "scenario": 1,
            "metrics": {
                metric: _cell(values, baseline_values, borrowed)
                for metric in ("land_use", "woodland", "urban")
            },
        }
    ]


class TestDetectingAnInertPerturbation:
    def test_identical_per_seed_values_are_inert(self) -> None:
        assert inert_perturbation(_scenarios([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])) is True

    def test_one_moved_seed_is_enough_to_be_informative(self) -> None:
        assert inert_perturbation(_scenarios([0.1, 0.2, 0.3], [0.1, 0.2, 0.4])) is False

    def test_a_tiny_move_still_counts_as_a_move(self) -> None:
        moved = [0.1, 0.2, 0.3 + 1e-12]
        assert inert_perturbation(_scenarios(moved, [0.1, 0.2, 0.3])) is False

    def test_a_borrowed_row_is_not_evidence_either_way(self) -> None:
        # Every row borrowed from the baseline campaign is equal to it by
        # construction, so a block of nothing but borrowed rows has compared
        # nothing and cannot be called inert.
        assert (
            inert_perturbation(_scenarios([0.1], [0.1], borrowed=True)) is False
        ), "a block with no re-run method compares nothing"

    def test_a_missing_baseline_is_not_evidence_either_way(self) -> None:
        assert inert_perturbation(_scenarios([0.1, 0.2, 0.3], None)) is False

    def test_a_block_with_no_scenarios_is_not_inert(self) -> None:
        assert inert_perturbation([]) is False


class TestCountingTheInformativeOnes:
    def _blocks(self) -> list[dict[str, Any]]:
        return [
            {"label": "a", "no_op": False, "inert": False, "scenarios": [{}]},
            {"label": "b", "no_op": False, "inert": True, "scenarios": [{}]},
            {"label": "c", "no_op": True, "inert": False, "scenarios": []},
        ]

    def test_only_effective_and_moving_perturbations_are_informative(self) -> None:
        assert [b["label"] for b in informative_perturbations(self._blocks())] == ["a"]

    def test_the_reason_names_what_happened(self) -> None:
        assert "identical" in INERT_REASON


class TestTheHeadlineSeparatesThemOut:
    def _overall(self) -> dict[str, Any]:
        return {
            "n_perturbations": 4,
            "n_effective": 3,
            "n_no_op": 1,
            "n_inert": 1,
            "n_informative": 2,
            "all_metrics_preserved": 3,
            "all_metrics_preserved_informative": 2,
            "metrics": {
                metric: {
                    "preserved": 3,
                    "total": 3,
                    "informative_preserved": 2,
                    "n_informative": 2,
                    "mean_kendall_tau": 1.0,
                    "cells_preserved": 3,
                    "n_cells": 3,
                    "absolute": {
                        "preserved": 3,
                        "total": 3,
                        "informative_preserved": 2,
                        "n_informative": 2,
                        "mean_kendall_tau": 1.0,
                        "cells_preserved": 3,
                        "n_cells": 3,
                    },
                }
                for metric in ("land_use", "woodland", "urban")
            },
        }

    def test_the_headline_counts_only_informative_perturbations(self) -> None:
        text = "\n".join(overall_section(self._overall()))
        assert "**2 / 2**" in text, text

    def test_the_headline_says_how_many_changed_nothing(self) -> None:
        text = "\n".join(overall_section(self._overall()))
        assert "1" in text and "identical" in text

    def test_a_headline_with_nothing_informative_says_so(self) -> None:
        overall = self._overall()
        overall.update(n_inert=3, n_informative=0, all_metrics_preserved_informative=0)
        for entry in overall["metrics"].values():
            entry["n_informative"] = 0
            entry["informative_preserved"] = 0
            entry["absolute"]["n_informative"] = 0
        text = "\n".join(overall_section(overall))
        assert "no perturbation" in text.lower()

    def test_an_inert_row_is_marked_in_the_verdict_table(self) -> None:
        blocks = [
            {
                "label": "replay_capacity_plus",
                "value_text": "120000",
                "no_op": False,
                "inert": True,
                "scenarios": _scenarios([0.1], [0.1]),
            }
        ]
        text = "\n".join(verdict_table(blocks))
        assert "replay_capacity_plus" in text
        assert "identical" in text.lower()


class TestTheEndToEndPayloadCarriesIt:
    """Pinned against the module-scoped tiny analysis in the driver tests."""

    def test_every_block_declares_whether_it_moved_anything(
        self, tiny_hparam_summary: dict[str, Any]
    ) -> None:
        for block in tiny_hparam_summary["perturbations"]:
            assert isinstance(block["inert"], bool)

    def test_every_compared_method_carries_the_baseline_per_seed_values(
        self, tiny_hparam_summary: dict[str, Any]
    ) -> None:
        for block in tiny_hparam_summary["perturbations"]:
            for scenario in block["scenarios"]:
                for cell in scenario["metrics"].values():
                    for entry in cell["methods"].values():
                        assert "baseline_values" in entry
                        if entry["baseline_values"] is not None:
                            assert len(entry["baseline_values"]) == len(entry["values"])

    def test_the_overall_block_counts_the_inert_ones(
        self, tiny_hparam_summary: dict[str, Any]
    ) -> None:
        overall = tiny_hparam_summary["overall"]
        assert overall["n_informative"] + overall["n_inert"] == overall["n_effective"]
        assert overall["all_metrics_preserved_informative"] <= overall["n_informative"]

    def test_a_three_iteration_run_cannot_detune_the_learner(
        self, tiny_hparam_summary: dict[str, Any]
    ) -> None:
        # The fixture runs three iterations, so the learner never reaches
        # ``learning_starts`` and a gamma move cannot reach the landscape.  The
        # payload has to say that rather than score it as a survival.
        gamma = [
            block
            for block in tiny_hparam_summary["perturbations"]
            if block["hyperparameter"] == "gamma"
        ]
        assert gamma, "the tiny analysis perturbs gamma"
        assert all(block["inert"] for block in gamma)

    def test_the_report_names_the_inert_perturbations(
        self, tiny_hparam_report: str
    ) -> None:
        assert "gamma_minus" in tiny_hparam_report
        assert "identical" in tiny_hparam_report.lower()
