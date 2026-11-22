"""The calibration verdict has to say what its p-value can and cannot carry.

An evaluation rollout produces tens of thousands of appraisals, and they are
not independent: 50 of them share one state, and consecutive iterations share a
landscape.  A Kolmogorov-Smirnov p-value under those two conditions rejects
almost any departure and is not a valid test of anything.  The KS *statistic*
is still a perfectly good effect size, so the package reports the boolean the
verdict needs and, beside it, the reason the boolean must not be read as a
significance test.  These tests pin that it is said, and said in both
directions.
"""

from __future__ import annotations

import numpy as np

from chapter4_qolu.diagnostics.records import AppraisalLog, DiagnosticsLog
from chapter4_qolu.diagnostics.report import render_diagnostics_report
from chapter4_qolu.diagnostics.summary import (
    CALIBRATION_CAVEAT,
    diagnostics_summary,
    verdict_for,
)


def _headline(ks_p: float, skill: float = 0.1) -> dict:
    return {
        "n": 30_000,
        "mae": 1.0,
        "rmse": 2.0,
        "bias": 0.0,
        "sd_realised": 3.0,
        "pit": {"ks_statistic": 0.02, "ks_p_value": ks_p},
        "crps": {"skill": {"climatology": skill, "initialisation": 0.5}},
        "boundary_mass": {"total_mean": 0.01, "threshold": 0.05, "flagged": False},
        "support": {
            "delta_z": 8.0,
            "realised_sd": 41.0,
            "realised_sd_in_atoms": 5.1,
            "distinct_atoms_used": 23,
            "n_atoms": 51,
            "resolves_the_signal": True,
        },
    }


def _log(rng: np.random.Generator) -> DiagnosticsLog:
    atoms = np.linspace(-200.0, 200.0, 51)
    probabilities = rng.random((40, 51))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return DiagnosticsLog(
        logs=(
            AppraisalLog(
                stakeholder=0,
                objective="land_use",
                atoms=atoms,
                probabilities=probabilities,
                reference_probabilities=probabilities.copy(),
                actions=rng.integers(0, 2, size=40),
                quality_scores=rng.normal(size=40),
                rewards=rng.normal(size=40),
                bootstrap_values=rng.normal(size=40),
                iteration_index=np.arange(40),
                gamma=0.99,
                n_step=3,
            ),
        ),
        iterations=4,
        samples_per_iteration=10,
        qolu_reward_mode="team",
        scenario_weights=(0.33, 0.33, 0.33),
    )


class TestTheCaveatIsStated:
    def test_the_verdict_carries_the_sample_size_and_the_caveat(self) -> None:
        verdict = verdict_for(_headline(ks_p=0.0))
        assert verdict["n"] == 30_000
        assert verdict["calibration_caveat"] == CALIBRATION_CAVEAT
        assert "not independent" in CALIBRATION_CAVEAT

    def test_the_caveat_is_stated_when_the_test_passes_too(self) -> None:
        passing = verdict_for(_headline(ks_p=0.9))
        failing = verdict_for(_headline(ks_p=0.0))
        assert passing["calibrated"] is True
        assert failing["calibrated"] is False
        assert passing["calibration_caveat"] == failing["calibration_caveat"]

    def test_the_statement_names_the_ks_statistic_as_the_effect_size(self) -> None:
        statement = verdict_for(_headline(ks_p=0.0))["statement"]
        assert "KS = 0.020" in statement

    def test_the_summary_passes_the_caveat_through(self) -> None:
        rng = np.random.default_rng(0)
        summary = diagnostics_summary(_log(rng), rng)
        assert summary["verdict"]["calibration_caveat"] == CALIBRATION_CAVEAT
        assert summary["verdict"]["n"] > 0

    def test_the_report_prints_the_caveat_and_the_sample_size(self) -> None:
        rng = np.random.default_rng(0)
        report = render_diagnostics_report(diagnostics_summary(_log(rng), rng))
        assert CALIBRATION_CAVEAT.split(".")[0] in report
        assert "appraisals scored" in report
