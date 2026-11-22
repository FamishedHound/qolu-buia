"""The thesis tables are views of ``summary.json`` and the diagnostics sweep.

They must show the view the thesis reports, mark exactly the tests the
aggregation computed on that view, bold the best value in the right direction
(``C_align`` is lower-is-better, Conversions has no best), use booktabs and no
colour, and come with a note saying what every number is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chapter4_qolu.make_thesis_tables import write_thesis_tables
from chapter4_qolu.reporting.aggregate import aggregate_records
from chapter4_qolu.reporting.thesis_tables import (
    distribution_groups,
    render_distribution_note,
    render_distribution_table,
    render_scenario_note,
    render_scenario_table,
    scenario_cells,
)
from chapter4_qolu.tests.test_reporting_aggregate import make_record

METHODS = ("no_regulator", "ltos", "pmaddpg", "mitm_learned", "qolu_buia")
#: (absolute land use, c_align) per method and seed: MitM (learned) has the
#: highest land use, QOLU+BUIA the lowest c_align.
VALUES = {
    "no_regulator": ([0.70, 0.72, 0.71], [0.40, 0.41, 0.42]),
    "ltos": ([0.80, 0.83, 0.81], [0.35, 0.36, 0.33]),
    "pmaddpg": ([0.85, 0.86, 0.88], [0.30, 0.31, 0.33]),
    "mitm_learned": ([0.95, 0.96, 0.94], [0.20, 0.22, 0.25]),
    "qolu_buia": ([0.91, 0.94, 0.93], [0.10, 0.12, 0.11]),
}


def campaign_records() -> list[dict[str, Any]]:
    out = []
    for method in METHODS:
        land_use, c_align = VALUES[method]
        for seed in range(3):
            record = make_record(
                method,
                seed,
                {"land_use": 0.5 + 0.01 * seed, "c_align": c_align[seed]},
                absolute={"land_use": land_use[seed]},
            )
            record["total_conversions"] = 1000 if method != "qolu_buia" else 780 + seed
            out.append(record)
    return out


@pytest.fixture()
def summary() -> dict[str, Any]:
    return aggregate_records(campaign_records(), metadata={"tag": "thesis"})


def _rows(text: str) -> dict[str, list[str]]:
    rows = {}
    for line in text.splitlines():
        line = line.strip()
        if line.endswith("\\\\") and "&" in line and not line.startswith("Method"):
            cells = [cell.strip() for cell in line[:-2].split("&")]
            rows[cells[0]] = cells[1:]
    return rows


class TestTheScenarioTable:
    def test_rows_follow_the_table_order_with_the_ablation_first(
        self, summary: dict[str, Any]
    ) -> None:
        text = render_scenario_table(summary, scenario_cells(summary)[0])
        assert list(_rows(text)) == [
            "No regulator",
            "LToS",
            "P-MADDPG",
            "MitM (learned)",
            "QOLU+BUIA",
        ]

    def test_columns_and_booktabs(self, summary: dict[str, Any]) -> None:
        text = render_scenario_table(summary, scenario_cells(summary)[0])
        assert (
            "Method & Land Use & Woodland & Urban & $C_{\\text{align}}$ & "
            "Conversions \\\\" in text
        )
        for rule in ("\\toprule", "\\midrule", "\\bottomrule"):
            assert text.count(rule) == 1
        assert "\\hline" not in text and "|" not in text.split("tabular}")[1][:6]
        assert "color" not in text.lower()

    def test_it_shows_the_absolute_view_by_default(
        self, summary: dict[str, Any]
    ) -> None:
        rows = _rows(render_scenario_table(summary, scenario_cells(summary)[0]))
        assert rows["No regulator"][0].startswith("$0.710 \\pm 0.006")

    def test_best_is_bold_in_the_right_direction(self, summary: dict[str, Any]) -> None:
        rows = _rows(render_scenario_table(summary, scenario_cells(summary)[0]))
        assert rows["MitM (learned)"][0].startswith("$\\mathbf{")  # land use, highest
        assert rows["QOLU+BUIA"][3].startswith("$\\mathbf{")  # c_align, lowest
        assert not any("mathbf" in row[4] for row in rows.values())  # conversions

    def test_the_marker_follows_the_stored_absolute_test(
        self, summary: dict[str, Any]
    ) -> None:
        cell = scenario_cells(summary)[0]
        rows = _rows(render_scenario_table(summary, cell))
        tests = cell["tests_by_view"]["absolute"]
        for method, label in (("no_regulator", "No regulator"), ("ltos", "LToS")):
            marked = rows[label][0].endswith("^{*}$")
            assert marked == (tests[method]["land_use"]["p_value"] < 0.05)
        assert not any("^{*}" in value for value in rows["QOLU+BUIA"])

    def test_conversions_are_one_decimal(self, summary: dict[str, Any]) -> None:
        rows = _rows(render_scenario_table(summary, scenario_cells(summary)[0]))
        assert rows["QOLU+BUIA"][4] == "$781.0 \\pm 0.6$"

    def test_a_summary_without_view_tests_is_refused(
        self, summary: dict[str, Any]
    ) -> None:
        cell = dict(scenario_cells(summary)[0])
        cell.pop("tests_by_view")
        with pytest.raises(ValueError, match="--aggregate"):
            render_scenario_table(summary, cell)

    def test_the_note_names_view_n_and_test(self, summary: dict[str, Any]) -> None:
        note = render_scenario_note(summary, scenario_cells(summary)[0])
        assert "absolute-normalised view (b)" in note
        assert "n = 3 seeds" in note
        assert "two-sided paired t-test" in note
        assert "No regulator, Land Use: mean difference" in note


def sweep_payload() -> dict[str, Any]:
    cells = []
    for scenario in (1, 2):
        for seed in range(3):
            cells.append(
                {
                    "tag": f"s{scenario}_seed{seed}",
                    "scenario": scenario,
                    "seed": seed,
                    "targets": {
                        "tail_return": {
                            "skill_climatology": -5.0 - seed,
                            "skill_initialisation": 0.4 + 0.01 * seed,
                            "ks_statistic": 0.8,
                            "coverage_50": 0.01,
                            "coverage_90": 0.5,
                            "mae": 80.0 + seed,
                        }
                    },
                    "policy": {
                        "stakeholders": [
                            {
                                "approval_rate": rate,
                                "degenerate_approval": rate in (0.0, 1.0),
                                "quality_score": {"within_iteration_sd": 0.007},
                            }
                            for rate in (1.0, 1.0, 0.5 if seed == 0 else 0.0)
                        ]
                    },
                }
            )
    return {
        "tag": "final",
        "headline_target": "tail_return",
        "n_cells": 6,
        "n_complete": 6,
        "settings": {"method": "qolu_buia", "scenarios": [1, 2], "seeds": [0, 1, 2]},
        "cells": cells,
    }


class TestTheDistributionTable:
    def test_groups_are_the_scenarios_then_pooled(self) -> None:
        names, summaries, target = distribution_groups(sweep_payload())
        assert names == ["Scenario 1", "Scenario 2", "Pooled"]
        assert summaries["Scenario 1"]["n"] == 3 and summaries["Pooled"]["n"] == 6
        assert target == "tail_return"

    def test_stakeholder_quantities_are_averaged_within_a_cell_first(self) -> None:
        _, summaries, _ = distribution_groups(sweep_payload())
        # per cell: seed 0 -> (1 + 1 + 0.5) / 3, seeds 1-2 -> (1 + 1 + 0) / 3
        expected = ((2.5 / 3) + (2 / 3) + (2 / 3)) / 3
        assert summaries["Scenario 1"]["approval_rate"].mean == pytest.approx(expected)
        # all or none: seed 0 has 2 of 3, seeds 1-2 have 3 of 3
        share = ((2 / 3) + 1.0 + 1.0) / 3
        assert summaries["Scenario 1"]["degenerate_share"].mean == pytest.approx(share)

    def test_the_table_has_every_requested_row(self) -> None:
        text = render_distribution_table(sweep_payload())
        for label in (
            "CRPS skill vs climatology",
            "CRPS skill vs initialisation",
            "PIT KS statistic",
            "central 50",
            "central 90",
            "MAE of",
            "Quality-score SD within an iteration",
            "approving all or none",
            "Greedy approval rate",
        ):
            assert label in text
        assert "Statistic & Scenario 1 & Scenario 2 & Pooled \\\\" in text
        assert "\\toprule" in text and "\\bottomrule" in text

    def test_the_note_states_the_target_and_the_sample(self) -> None:
        note = render_distribution_note(sweep_payload())
        assert "tail_return" in note
        assert "Scenario 1: 3" in note and "Pooled: 6" in note


class TestTheCommand:
    def test_it_writes_a_table_and_a_note_per_scenario_and_for_the_distribution(
        self, summary: dict[str, Any], tmp_path: Path
    ) -> None:
        campaign = tmp_path / "campaign_x"
        campaign.mkdir()
        (campaign / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        diagnostics = tmp_path / "diag"
        diagnostics.mkdir()
        (diagnostics / "distribution_diagnostics_sweep.json").write_text(
            json.dumps(sweep_payload()), encoding="utf-8"
        )
        written = write_thesis_tables(campaign, diagnostics)
        names = sorted(path.name for path in written)
        assert names == [
            "table_distribution.tex",
            "table_distribution.txt",
            "table_scenario1.tex",
            "table_scenario1.txt",
        ]
        assert all(path.parent == campaign / "thesis" for path in written)


class TestTheLearnerIsNamed:
    def test_the_note_states_the_learner_it_was_given(self) -> None:
        note = render_distribution_note(sweep_payload(), learner=(25, [39, 39, 39]))
        assert "t_n = 25" in note
        assert "39 target-network syncs per stakeholder" in note

    def test_the_command_reads_t_n_off_the_cell_summaries(
        self, summary: dict[str, Any], tmp_path: Path
    ) -> None:
        campaign = tmp_path / "campaign_y"
        campaign.mkdir()
        (campaign / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        payload = sweep_payload()
        for index, cell in enumerate(payload["cells"]):
            path = tmp_path / f"cell{index}.json"
            path.write_text(
                json.dumps(
                    {
                        "learner": {
                            "target_sync_interval": 25,
                            "target_syncs": [39, 39, 39],
                        }
                    }
                ),
                encoding="utf-8",
            )
            cell["summary_path"] = str(path)
        diagnostics = tmp_path / "diag"
        diagnostics.mkdir()
        (diagnostics / "distribution_diagnostics_sweep.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        write_thesis_tables(campaign, diagnostics)
        note = (campaign / "thesis" / "table_distribution.txt").read_text(
            encoding="utf-8"
        )
        assert "t_n = 25" in note


class TestTheNoteRelatesCAlignToConversions:
    """With whole-parcel conversion, C_align is close to conversions / |P|.

    Each conversion turns one non-urban parcel urban, which moves the
    neighbourhood distribution of every adjacent parcel by exactly 1/n (n its
    neighbour count) whatever the land types were, so the note states the
    observed ratio rather than leaving the reader to infer it.
    """

    def test_the_note_states_the_ratio_range(self, summary: dict[str, Any]) -> None:
        configuration = summary["configurations"][0]
        cell = configuration["scenarios"][0]
        note = render_scenario_note(summary, cell, configuration=configuration)
        ratios = [
            c * 900 / n
            for method in METHODS
            for c, n in zip(
                cell["methods"][method]["absolute"]["c_align"]["values"],
                cell["methods"][method]["total_conversions"]["values"],
            )
        ]
        assert "C_align x |P| / Conversions" in note
        assert f"{min(ratios):.3f} to {max(ratios):.3f}" in note
        assert "|P| = 900" in note

    def test_without_the_configuration_the_line_is_left_out(
        self, summary: dict[str, Any]
    ) -> None:
        note = render_scenario_note(summary, scenario_cells(summary)[0])
        assert "C_align x |P| / Conversions" not in note
