"""How many *independent* observations a diagnostics statistic really has.

An evaluation rollout logs one appraisal per candidate and 50 candidates per
iteration.  Under the ``objective`` and ``team`` readings of ``R_QOLU,i`` the
reward a stakeholder is paid is one scalar for the whole synchronised update
(:func:`chapter4_qolu.runner._transition_rewards`), so all 50 appraisals of an
iteration are scored against the *same* realised return and the nominal sample
size of 10 000 is really 200.  Every statistic here is still correct; what
changes is how many of them a reader is entitled to call significant.  These
tests pin that the package measures and reports that, rather than leaving it to
prose.
"""

from __future__ import annotations

import numpy as np

from chapter4_qolu.diagnostics.errors import target_clustering


class TestConstantWithinIteration:
    def test_a_team_style_target_has_one_observation_per_iteration(self) -> None:
        iterations = np.repeat(np.arange(4), 5)
        realised = np.repeat(np.array([1.0, 2.0, 3.0, 4.0]), 5)
        block = target_clustering(iterations, realised)
        assert block["n_appraisals"] == 20
        assert block["n_iterations"] == 4
        assert block["constant_within_iteration"] is True
        assert block["effective_n"] == 4
        assert block["appraisals_per_iteration"] == 5.0

    def test_a_per_parcel_target_keeps_its_full_sample(self) -> None:
        iterations = np.repeat(np.arange(4), 5)
        realised = np.arange(20, dtype=np.float64)
        block = target_clustering(iterations, realised)
        assert block["constant_within_iteration"] is False
        assert block["effective_n"] == 20

    def test_distinct_values_are_counted(self) -> None:
        iterations = np.repeat(np.arange(3), 2)
        realised = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
        block = target_clustering(iterations, realised)
        assert block["distinct_realised_values"] == 2
        assert block["constant_within_iteration"] is True
        assert block["effective_n"] == 3

    def test_a_single_iteration_is_one_effective_observation(self) -> None:
        block = target_clustering(np.zeros(50, dtype=np.int64), np.ones(50))
        assert block["n_iterations"] == 1
        assert block["effective_n"] == 1

    def test_floating_point_noise_within_an_iteration_still_counts_as_constant(
        self,
    ) -> None:
        iterations = np.repeat(np.arange(2), 3)
        realised = np.array([1.0, 1.0 + 1e-15, 1.0, 5.0, 5.0, 5.0 - 1e-15])
        block = target_clustering(iterations, realised)
        assert block["constant_within_iteration"] is True


class TestItIsWiredIntoTheSummary:
    def test_every_target_block_carries_its_clustering(self) -> None:
        from chapter4_qolu.diagnostics.records import AppraisalLog
        from chapter4_qolu.diagnostics.summary import target_statistics

        rng = np.random.default_rng(3)
        atoms = np.linspace(-200.0, 200.0, 51)
        probabilities = rng.random((20, 51))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        log = AppraisalLog(
            stakeholder=0,
            objective="land_use",
            atoms=atoms,
            probabilities=probabilities,
            reference_probabilities=probabilities.copy(),
            actions=rng.integers(0, 2, size=20),
            quality_scores=rng.normal(size=20),
            rewards=np.repeat(np.array([1.0, 2.0, 3.0, 4.0]), 5),
            bootstrap_values=np.zeros(20),
            iteration_index=np.repeat(np.arange(4), 5),
            gamma=0.99,
            n_step=3,
        )
        statistics = target_statistics(log, log.rewards, rng)
        assert statistics["clustering"]["effective_n"] == 4
        assert statistics["clustering"]["constant_within_iteration"] is True


class TestItReachesTheVerdictAndTheDocuments:
    @staticmethod
    def _headline(effective: int, nominal: int = 10_000) -> dict:
        return {
            "n": nominal,
            "mae": 1.0,
            "rmse": 2.0,
            "bias": 0.0,
            "mean_predicted": 0.5,
            "mean_realised": 0.5,
            "sd_realised": 3.0,
            "pit": {
                "ks_statistic": 0.2,
                "ks_p_value": 0.0,
                "out_of_support_fraction": 0.0,
            },
            "coverage": [
                {"level": 0.5, "empirical": 0.4, "mean_width": 10.0},
                {"level": 0.9, "empirical": 0.95, "mean_width": 40.0},
            ],
            "crps": {
                "model": 1.0,
                "reference": {"initialisation": 2.0, "climatology": 1.1},
                "skill": {"climatology": 0.1, "initialisation": 0.5},
            },
            "boundary_mass": {"total_mean": 0.01, "threshold": 0.05, "flagged": False},
            "support": {
                "delta_z": 8.0,
                "realised_sd": 41.0,
                "realised_sd_in_atoms": 5.1,
                "distinct_atoms_used": 23,
                "n_atoms": 51,
                "resolves_the_signal": True,
            },
            "clustering": {
                "n_appraisals": nominal,
                "n_iterations": effective,
                "appraisals_per_iteration": nominal / effective,
                "distinct_realised_values": effective,
                "constant_within_iteration": True,
                "effective_n": effective,
            },
        }

    def test_the_verdict_reports_the_effective_sample_size(self) -> None:
        from chapter4_qolu.diagnostics.summary import verdict_for

        verdict = verdict_for(self._headline(effective=200))
        assert verdict["effective_n"] == 200
        assert "200" in verdict["statement"]

    def test_a_varying_target_still_reports_how_many_values_it_has(self) -> None:
        """Pooling three stakeholders can treble ``n`` without a new observation.

        Under the ``team`` reading every stakeholder is paid the same scalar, so
        the pooled realised-return column is three identical copies of one
        stream.  The distinct count is the number that shows it.
        """
        from chapter4_qolu.diagnostics.summary import verdict_for

        headline = self._headline(effective=200, nominal=30_000)
        headline["clustering"] = {
            "n_appraisals": 30_000,
            "n_iterations": 200,
            "appraisals_per_iteration": 150.0,
            "distinct_realised_values": 10_000,
            "constant_within_iteration": False,
            "effective_n": 30_000,
        }
        statement = verdict_for(headline)["statement"]
        assert "10000" in statement
        assert "200" in statement

    def test_a_verdict_without_clustering_still_renders(self) -> None:
        from chapter4_qolu.diagnostics.summary import verdict_for

        headline = self._headline(effective=200)
        headline.pop("clustering")
        verdict = verdict_for(headline)
        assert verdict["effective_n"] == 10_000

    def test_the_per_cell_report_prints_the_effective_sample_size(self) -> None:
        import numpy as np

        from chapter4_qolu.diagnostics.records import AppraisalLog, DiagnosticsLog
        from chapter4_qolu.diagnostics.report import render_diagnostics_report
        from chapter4_qolu.diagnostics.summary import diagnostics_summary

        rng = np.random.default_rng(5)
        atoms = np.linspace(-200.0, 200.0, 51)
        probabilities = rng.random((20, 51))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        log = DiagnosticsLog(
            logs=(
                AppraisalLog(
                    stakeholder=0,
                    objective="land_use",
                    atoms=atoms,
                    probabilities=probabilities,
                    reference_probabilities=probabilities.copy(),
                    actions=rng.integers(0, 2, size=20),
                    quality_scores=rng.normal(size=20),
                    rewards=np.repeat(np.array([1.0, 2.0, 3.0, 4.0]), 5),
                    bootstrap_values=np.zeros(20),
                    iteration_index=np.repeat(np.arange(4), 5),
                    gamma=0.99,
                    n_step=3,
                ),
            ),
            iterations=4,
            samples_per_iteration=5,
            qolu_reward_mode="team",
            scenario_weights=(0.33, 0.33, 0.33),
        )
        report = render_diagnostics_report(diagnostics_summary(log, rng))
        assert "independent observations" in report

    def test_the_sweep_row_carries_the_effective_sample_size(self) -> None:
        from chapter4_qolu.diagnostics.sweep_summary import flatten_target

        flat = flatten_target(self._headline(effective=200))
        assert flat["effective_n"] == 200
        assert flat["n"] == 10_000
