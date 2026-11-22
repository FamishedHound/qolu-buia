"""Turning the perturbed campaigns on disk into the sensitivity payload.

The claim under test - "the robustness of comparative conclusions" - is about
the **ordering** of the four methods, so every cell of the payload carries
three things: the per-method ``mean +/- SE`` over seeds, the ranking those
means produce, and how that ranking compares with the baseline configuration's
(identical or not, and Kendall's tau).

Design note (which view the ranking is read off).  The campaign carries three
readings of the same metrics.  ``minmax`` is an increasing affine map of
``raw`` within a cell, so it **cannot** order the methods differently and is
not reported separately here; ``absolute`` clips distances at a physical
saturation distance before averaging and genuinely can, so its ranking verdict
is carried beside the ``raw`` one in every cell and in the overall counts.

Design note (how a baseline method reaches an agent-side row).  An agent-side
perturbation re-runs only the QOLU learner (see
:mod:`chapter4_qolu.hparam_plan`); the other three methods' records would be
bit-identical to the baseline campaign's, so they are read from there and each
one is flagged ``borrowed_from_baseline`` in the payload rather than silently
presented as a fresh run.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chapter4_qolu.hparam_counts import RANKING_VIEWS, overall_counts
from chapter4_qolu.hparam_inert import (
    INERT_REASON,
    inert_perturbation,
    informative_perturbations,
)
from chapter4_qolu.hparam_plan import (
    BASELINE_TAG,
    HParamSettings,
    hparam_paths,
    perturbation_methods,
    perturbation_runs,
    settings_perturbations,
)
from chapter4_qolu.hparam_sensitivity import (
    Perturbation,
    kendall_tau,
    resolve_hyperparameters,
)
from chapter4_qolu.reporting.aggregate import (
    PUBLISHED_METRICS,
    aggregate_records,
    load_run_records,
    rank_methods,
)
from chapter4_qolu.reporting.stats import paired_t_test

__all__ = [
    "INERT_REASON",
    "RANKING_VIEWS",
    "build_hparam_summary",
    "campaign_records",
    "every_record",
    "inert_perturbation",
    "informative_perturbations",
    "overall_counts",
]


def _campaign_dir(settings: HParamSettings, tag: str) -> Path:
    return hparam_paths(settings).runs / f"campaign_{tag}"


def campaign_records(settings: HParamSettings, tag: str) -> list[dict[str, Any]]:
    """Every finished record of one campaign, or an empty list if there is none."""
    directory = _campaign_dir(settings, tag)
    if not directory.is_dir():
        return []
    return load_run_records(directory)


def every_record(settings: HParamSettings) -> list[dict[str, Any]]:
    """Every record the analysis has written, baseline first, in plan order."""
    records = list(campaign_records(settings, BASELINE_TAG))
    for perturbation in settings_perturbations(settings):
        records.extend(campaign_records(settings, perturbation.label))
    return records


def _scenario_blocks(records: Sequence[Mapping[str, Any]]) -> dict[int, Any]:
    """Aggregate a set of records into ``{scenario: aggregate block}``."""
    if not records:
        return {}
    summary = aggregate_records(records)
    blocks: dict[int, Any] = {}
    for configuration in summary["configurations"]:
        for scenario in configuration["scenarios"]:
            blocks[int(scenario["scenario"])] = scenario
    return blocks


def _ranking(methods: Mapping[str, Any], view: str, metric: str) -> tuple[str, ...]:
    means = {
        name: float(block[view][metric]["mean"])
        for name, block in methods.items()
        if metric in block.get(view, {})
    }
    return rank_methods(means) if means else ()


def _paired_against_baseline(
    perturbed: Mapping[str, Any], baseline: Mapping[str, Any], metric: str
) -> dict[str, Any]:
    """The method's own score under the perturbation against under the baseline.

    Paired on the seed, which is what makes the comparison meaningful: the same
    seed drives the same crop and the same proposal stream under both.
    """
    left = dict(zip(perturbed.get("seeds", []), perturbed["raw"][metric]["values"]))
    right = dict(zip(baseline.get("seeds", []), baseline["raw"][metric]["values"]))
    shared = sorted(set(left) & set(right))
    return paired_t_test(
        [left[seed] for seed in shared], [right[seed] for seed in shared]
    ).as_record()


def _aligned_baseline_values(
    perturbed: Mapping[str, Any], baseline: Mapping[str, Any], metric: str
) -> list[float] | None:
    """The baseline's per-seed values in the perturbed run's own seed order.

    ``None`` when a seed of the perturbed block has no baseline counterpart, so
    that a partial comparison is never silently padded into a full one.
    """
    right = dict(zip(baseline.get("seeds", []), baseline["raw"][metric]["values"]))
    seeds = list(perturbed.get("seeds", []))
    if not seeds or any(seed not in right for seed in seeds):
        return None
    return [float(right[seed]) for seed in seeds]


def _method_entry(
    block: Mapping[str, Any],
    baseline_block: Mapping[str, Any] | None,
    metric: str,
    borrowed: bool,
) -> dict[str, Any]:
    raw = block["raw"][metric]
    entry: dict[str, Any] = {
        "mean": float(raw["mean"]),
        "sd": float(raw["sd"]),
        "se": float(raw["se"]),
        "n": int(raw["n"]),
        "values": list(raw.get("values", [])),
        "seeds": list(block.get("seeds", [])),
        "borrowed_from_baseline": bool(borrowed),
        "baseline_mean": None,
        # The baseline's own per-seed vector, aligned seed for seed with
        # ``values``: the only thing that can distinguish "the ranking survived
        # the move" from "the move never reached the run" (:mod:`hparam_inert`).
        "baseline_values": None,
        "delta": None,
        "paired": None,
    }
    if baseline_block is not None and metric in baseline_block.get("raw", {}):
        baseline_mean = float(baseline_block["raw"][metric]["mean"])
        entry["baseline_mean"] = baseline_mean
        entry["baseline_values"] = _aligned_baseline_values(
            block, baseline_block, metric
        )
        entry["delta"] = entry["mean"] - baseline_mean
        entry["paired"] = _paired_against_baseline(block, baseline_block, metric)
    return entry


def _metric_cell(
    methods: Mapping[str, Any],
    baseline_methods: Mapping[str, Any],
    metric: str,
    borrowed: Sequence[str],
) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "methods": {
            name: _method_entry(
                block,
                baseline_methods.get(name),
                metric,
                borrowed=name in borrowed,
            )
            for name, block in sorted(methods.items())
            if metric in block.get("raw", {})
        }
    }
    for view in RANKING_VIEWS:
        ranking = _ranking(methods, view, metric)
        baseline_ranking = _ranking(baseline_methods, view, metric)
        comparable = bool(ranking) and sorted(ranking) == sorted(baseline_ranking)
        verdict = {
            "ranking": list(ranking),
            "baseline_ranking": list(baseline_ranking),
            "ranking_preserved": comparable
            and tuple(ranking) == tuple(baseline_ranking),
            "kendall_tau": (
                kendall_tau(ranking, baseline_ranking) if comparable else None
            ),
        }
        if view == "raw":
            cell.update(verdict)
        else:
            cell[view] = verdict
    return cell


def _perturbation_block(
    settings: HParamSettings,
    perturbation: Perturbation,
    baseline_blocks: Mapping[int, Any],
) -> dict[str, Any]:
    rerun = perturbation_methods(settings, perturbation)
    borrowed = tuple(name for name in settings.methods if name not in rerun)
    records = campaign_records(settings, perturbation.label)
    baseline_records = campaign_records(settings, BASELINE_TAG)
    merged = list(records) + [
        record for record in baseline_records if str(record["method"]) in borrowed
    ]
    blocks = _scenario_blocks(merged)

    block = perturbation.as_record()
    block.update(
        {
            "methods_rerun": list(rerun),
            "methods_borrowed": list(borrowed),
            "n_records": len(records),
            "n_planned": len(perturbation_runs(settings, perturbation)),
            "complete": len(records) == len(perturbation_runs(settings, perturbation)),
            "scenarios": [
                {
                    "scenario": scenario,
                    "metrics": {
                        metric: _metric_cell(
                            blocks[scenario]["methods"],
                            baseline_blocks.get(scenario, {}).get("methods", {}),
                            metric,
                            borrowed,
                        )
                        for metric in PUBLISHED_METRICS
                    },
                }
                for scenario in sorted(blocks)
            ],
        }
    )
    if perturbation.is_no_op:
        block["scenarios"] = []
        block["why_not_run"] = (
            "rounding or clamping put the value back on the baseline, so this "
            "perturbation is the identity and was not run"
        )
    block["inert"] = inert_perturbation(block["scenarios"])
    if block["inert"]:
        block["why_inert"] = INERT_REASON
    return block


def _hyperparameter_records(settings: HParamSettings) -> list[dict[str, Any]]:
    """One row per hyper-parameter, with both directions side by side."""
    chosen = resolve_hyperparameters(settings.hyperparameter_names)
    rows: list[dict[str, Any]] = []
    for item in chosen:
        row = item.as_record()
        for perturbation in settings_perturbations(settings):
            if perturbation.name == item.name:
                row[perturbation.direction_label] = perturbation.as_record()
        rows.append(row)
    return rows


def build_hparam_summary(settings: HParamSettings) -> dict[str, Any]:
    """The whole payload, read from the records on disk and nothing else."""
    baseline_records = campaign_records(settings, BASELINE_TAG)
    baseline_blocks = _scenario_blocks(baseline_records)
    blocks = [
        _perturbation_block(settings, perturbation, baseline_blocks)
        for perturbation in settings_perturbations(settings)
    ]
    n_no_op = sum(1 for block in blocks if block["no_op"])
    return {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tag": settings.tag,
        "promise": (
            "Sensitivity analyses are conducted by perturbing key "
            "hyperparameters (typically +/-20%) to assess the robustness of "
            "comparative conclusions."
        ),
        "settings": settings.as_record(),
        "views": {
            "reported": list(RANKING_VIEWS),
            "note": (
                "Rankings are read off the RAW view; the absolute-normalised "
                "verdict is carried beside it.  The cross-method min-max view "
                "is an increasing affine map of RAW within a cell, so it "
                "cannot order the methods differently and is not repeated."
            ),
        },
        "hyperparameters": _hyperparameter_records(settings),
        "baseline": {
            "tag": BASELINE_TAG,
            "n_records": len(baseline_records),
            "scenarios": [
                {
                    "scenario": scenario,
                    "metrics": {
                        metric: {
                            "ranking": list(_ranking(block["methods"], "raw", metric)),
                            "methods": {
                                name: {
                                    "mean": float(entry["raw"][metric]["mean"]),
                                    "se": float(entry["raw"][metric]["se"]),
                                    "n": int(entry["raw"][metric]["n"]),
                                }
                                for name, entry in sorted(block["methods"].items())
                                if metric in entry.get("raw", {})
                            },
                        }
                        for metric in PUBLISHED_METRICS
                    },
                }
                for scenario, block in sorted(baseline_blocks.items())
            ],
        },
        "perturbations": blocks,
        "overall": overall_counts(blocks, n_no_op),
    }
