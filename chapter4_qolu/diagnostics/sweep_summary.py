"""Read the per-cell diagnostics summaries back and aggregate across them.

One diagnostics run answers "is this seed's return distribution any good?".  A
chapter claim needs "is the *method's* return distribution any good?", and the
two differ by the seed variance, which a single run cannot show.  This module
flattens each cell's headline statistics into scalars, then reports them with a
mean, a sample standard deviation and a standard error of the mean over the
cells - overall, per scenario and per stakeholder objective.

Design note (no significance test): the cells differ in scenario *and* seed and
there is no comparator to pair against, so the aggregation reports spread and
counts rather than a p-value.  The per-cell booleans are counted, never
averaged: "9 of 12 cells are calibrated" is a statement a reader can check,
whereas a mean of four booleans is not.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from chapter4_qolu.diagnostics.plan import DiagnosticsSettings, diagnostics_paths
from chapter4_qolu.diagnostics.records import HEADLINE_TARGET
from chapter4_qolu.diagnostics.returns import TARGET_NAMES
from chapter4_qolu.diagnostics.summary import CALIBRATION_CAVEAT
from chapter4_qolu.reporting.stats import summarise

__all__ = [
    "AGGREGATED_FIELDS",
    "BOOLEAN_FIELDS",
    "aggregate_records",
    "cell_record",
    "flatten_target",
    "per_objective_records",
    "per_scenario_records",
    "sweep_verdict",
]

#: The scalars averaged across cells, in reporting order.
AGGREGATED_FIELDS: tuple[str, ...] = (
    "n",
    "effective_n",
    "mae",
    "rmse",
    "bias",
    "sd_realised",
    "ks_statistic",
    "ks_p_value",
    "coverage_50",
    "width_50",
    "coverage_90",
    "width_90",
    "crps",
    "crps_climatology",
    "crps_initialisation",
    "skill_climatology",
    "skill_initialisation",
    "boundary_mass",
    "out_of_support_fraction",
    "realised_sd_in_atoms",
    "distinct_atoms_used",
)

#: Per-cell yes/no answers.  Counted over the cells, never averaged.
BOOLEAN_FIELDS: tuple[str, ...] = (
    "state_conditional",
    "calibrated",
    "support_adequate",
    "resolves_the_signal",
)


def _coverage(target: dict[str, Any], level: float, key: str) -> float:
    for entry in target.get("coverage", ()):
        if abs(float(entry["level"]) - level) < 1e-9:
            return float(entry[key])
    return float("nan")


def flatten_target(target: dict[str, Any]) -> dict[str, Any]:
    """The scalar statistics of one target block, flat enough to average."""
    crps = target["crps"]
    support = target["support"]
    pit = target["pit"]
    boundary = target["boundary_mass"]
    clustering = target.get("clustering", {})
    return {
        "n": int(target["n"]),
        "effective_n": int(clustering.get("effective_n", target["n"])),
        "constant_within_iteration": bool(
            clustering.get("constant_within_iteration", False)
        ),
        "mae": float(target["mae"]),
        "rmse": float(target["rmse"]),
        "bias": float(target["bias"]),
        "mean_predicted": float(target["mean_predicted"]),
        "mean_realised": float(target["mean_realised"]),
        "sd_realised": float(target["sd_realised"]),
        "ks_statistic": float(pit["ks_statistic"]),
        "ks_p_value": float(pit["ks_p_value"]),
        "out_of_support_fraction": float(pit["out_of_support_fraction"]),
        "coverage_50": _coverage(target, 0.5, "empirical"),
        "width_50": _coverage(target, 0.5, "mean_width"),
        "coverage_90": _coverage(target, 0.9, "empirical"),
        "width_90": _coverage(target, 0.9, "mean_width"),
        "crps": float(crps["model"]),
        "crps_climatology": float(crps["reference"]["climatology"]),
        "crps_initialisation": float(crps["reference"]["initialisation"]),
        "skill_climatology": float(crps["skill"]["climatology"]),
        "skill_initialisation": float(crps["skill"]["initialisation"]),
        "boundary_mass": float(boundary["total_mean"]),
        "boundary_flagged": bool(boundary["flagged"]),
        "delta_z": float(support["delta_z"]),
        "realised_sd_in_atoms": float(support["realised_sd_in_atoms"]),
        "distinct_atoms_used": int(support["distinct_atoms_used"]),
        "resolves_the_signal": bool(support["resolves_the_signal"]),
    }


def _flatten_all(targets: dict[str, Any]) -> dict[str, Any]:
    return {
        name: flatten_target(targets[name]) for name in TARGET_NAMES if name in targets
    }


def cell_record(settings: DiagnosticsSettings) -> dict[str, Any] | None:
    """One sweep row, or ``None`` when that cell has not finished.

    Nothing is recomputed here: the row is read out of the cell's own
    ``distribution_diagnostics.json``, so a sweep table and the per-cell
    document it points at can never disagree.
    """
    path = diagnostics_paths(settings).summary
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(summary, dict) or "verdict" not in summary:
        return None
    verdict = summary["verdict"]
    return {
        "tag": summary.get("tag", settings.tag),
        "scenario": int(settings.scenario),
        "seed": int(settings.seed),
        "headline_target": summary.get("headline_target", HEADLINE_TARGET),
        "agents": summary.get("agents", {}),
        "rollout": summary.get("rollout", {}),
        "targets": _flatten_all(summary["pooled"]["targets"]),
        "stakeholders": [
            {
                "stakeholder": int(entry["stakeholder"]),
                "objective": str(entry["objective"]),
                "approval_rate": float(entry.get("approval_rate", float("nan"))),
                "mean_reward": float(entry.get("mean_reward", float("nan"))),
                # ``{}`` for a cell whose summary carries no spread: the row
                # says "not measured" rather than inventing a zero, which
                # would read as "constant policy".
                "quality_score": dict(entry.get("quality_score", {})),
                "targets": _flatten_all(entry["targets"]),
            }
            for entry in summary.get("stakeholders", [])
        ],
        "policy": dict(summary.get("policy", {})),
        "state_conditional": bool(verdict["state_conditional"]),
        "calibrated": bool(verdict["calibrated"]),
        "support_adequate": bool(verdict["support_adequate"]),
        "resolves_the_signal": bool(verdict["support_resolves_the_signal"]),
        "summary_path": str(path),
    }


def _values(records: Sequence[dict[str, Any]], target: str, field: str) -> list[float]:
    out: list[float] = []
    for record in records:
        block = record["targets"].get(target)
        if block is not None and field in block:
            out.append(float(block[field]))
    return out


def aggregate_records(
    records: Sequence[dict[str, Any]], target: str = HEADLINE_TARGET
) -> dict[str, Any]:
    """Mean, SD, SE and the raw sample of every averaged field across cells."""
    if not records:
        return {}
    aggregate: dict[str, Any] = {}
    for field in AGGREGATED_FIELDS:
        sample = _values(records, target, field)
        if sample:
            aggregate[field] = summarise(sample).as_record()
    return aggregate


def per_scenario_records(
    records: Sequence[dict[str, Any]], target: str = HEADLINE_TARGET
) -> dict[str, Any]:
    """The same aggregation, split by scenario, keyed by its number as text."""
    scenarios = sorted({int(record["scenario"]) for record in records})
    return {
        str(scenario): aggregate_records(
            [item for item in records if int(item["scenario"]) == scenario], target
        )
        for scenario in scenarios
    }


def per_objective_records(
    records: Sequence[dict[str, Any]], target: str = HEADLINE_TARGET
) -> dict[str, Any]:
    """The same aggregation per stakeholder objective, across every cell."""
    rows: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for entry in record["stakeholders"]:
            rows.setdefault(entry["objective"], []).append(
                {"targets": entry["targets"]}
            )
    return {name: aggregate_records(rows[name], target) for name in sorted(rows)}


def _count(records: Iterable[dict[str, Any]], field: str) -> int:
    return sum(1 for record in records if bool(record[field]))


def sweep_verdict(
    records: Sequence[dict[str, Any]],
    overall: dict[str, Any],
    target: str = HEADLINE_TARGET,
) -> dict[str, Any]:
    """Counts over the cells and one paragraph, stated the same way either way."""
    total = len(records)
    counts = {field: _count(records, field) for field in BOOLEAN_FIELDS}
    skill = overall.get("skill_climatology")
    initialisation = overall.get("skill_initialisation")
    if total == 0 or skill is None:
        return {
            "n_cells": total,
            "n_state_conditional": 0,
            "n_calibrated": 0,
            "n_support_adequate": 0,
            "n_resolves_the_signal": 0,
            "calibration_caveat": CALIBRATION_CAVEAT,
            "statement": (
                "No diagnostics cell has finished, so nothing is claimed about "
                "the learnt return distributions."
            ),
        }

    unanimous = counts["state_conditional"] == total
    none_at_all = counts["state_conditional"] == 0
    direction = (
        "every cell"
        if unanimous
        else ("no cell" if none_at_all else f"{counts['state_conditional']} of {total}")
    )
    statement = (
        f"Across {total} cell(s) scored against `{target}`, CRPS skill against "
        f"climatology averages {skill['mean']:+.4f} "
        f"(SD {skill['sd']:.4f}, SE {skill['se']:.4f}); "
        f"{direction} beat(s) climatology, so "
        + (
            "the learnt distribution carries state-conditional information "
            "consistently across scenarios and seeds."
            if unanimous
            else (
                "no cell's distribution carries more information than the run's "
                "own marginal histogram of returns."
                if none_at_all
                else "the result is not consistent across scenarios and seeds "
                "and must be quoted cell by cell."
            )
        )
        + f"  CRPS skill against the network at initialisation averages "
        f"{initialisation['mean']:+.4f} (SD {initialisation['sd']:.4f}).  "
        f"{counts['calibrated']} of {total} cell(s) have a randomised PIT "
        f"consistent with uniformity, {counts['support_adequate']} of {total} "
        f"hold the distribution inside the atom support, and "
        f"{counts['resolves_the_signal']} of {total} have an atom spacing fine "
        "enough to express the realised variation."
    )
    return {
        "n_cells": total,
        "n_state_conditional": counts["state_conditional"],
        "n_calibrated": counts["calibrated"],
        "n_support_adequate": counts["support_adequate"],
        "n_resolves_the_signal": counts["resolves_the_signal"],
        "mean_skill_climatology": float(skill["mean"]),
        "mean_skill_initialisation": float(initialisation["mean"]),
        "consistent": bool(unanimous or none_at_all),
        "calibration_caveat": CALIBRATION_CAVEAT,
        "statement": statement,
    }
