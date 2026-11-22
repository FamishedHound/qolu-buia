"""Turning the sweep's run records into ``sensitivity_summary.json``.

Everything here reads what is on disk and returns plain JSON-ready data, so
the summary - and with it ``sensitivity.md`` - can be rebuilt from the records
alone, without running or even loading a single environment.  The same split
:mod:`chapter4_qolu.scan_summary` makes for the window scan.

The one thing worth spelling out is how a *configuration* finds its four runs.
Stage A runs 64 cells per window and a configuration selects four of them:

* QOLU + BUIA and the chosen P-MADDPG variant from the campaign of its **reward
  reading**, at its cap;
* the chosen LToS and MitM variants from the window's single **baselines**
  campaign, at the same cap - those two methods never see a reward reading, so
  running them once per mode would have produced identical records.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS, load_run_records
from chapter4_qolu.reporting.documents import build_summary, write_campaign_documents
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.reporting.published import (
    PUBLISHED_CLOSENESS_VIEW,
    normalisation_check,
    value_closeness,
)
from chapter4_qolu.scan_scoring import count_pattern_matches
from chapter4_qolu.sensitivity_plan import (
    BASELINE_CAMPAIGN_SUFFIX,
    SensitivityConfiguration,
    SensitivitySettings,
    StageBChoice,
)
from chapter4_qolu.sensitivity_scoring import (
    ConfigurationScore,
    best_per_configuration,
    count_pattern_matches_tie_aware,
    score_configuration,
)
from chapter4_qolu.sensitivity_stages import (
    stage_a_campaigns,
    stage_b_campaigns,
    stage_c_campaigns,
)
from chapter4_qolu.window_scan import ScanWindow

__all__ = [
    "StageAIndex",
    "campaign_value_mae",
    "load_stage_a_index",
    "published_record",
    "reference_records",
    "score_all_configurations",
    "select_stage_b_choices",
    "stage_a_record",
    "stage_b2_records",
    "stage_b2_windows_on_disk",
    "stage_b_records",
    "stage_c_record",
]

logger = logging.getLogger(__name__)

#: ``(window label, campaign kind, configuration label, method)`` -> record.
StageAIndex = dict[tuple[str, str, str, str], Mapping[str, Any]]


def campaign_kind(tag: str, window_label: str) -> str:
    """What a stage-A campaign tag says after the window: mode, or method.

    ``a_<window>_<reward mode>`` and ``a_<window>_baselines`` are the swept
    campaigns; ``ref_<window>_<method>`` is a documented reference run.  Both
    are indexed under whatever follows the window, so the lookup key is the
    same shape for either.
    """
    text = str(tag)
    for prefix in (f"a_{window_label}_", f"ref_{window_label}_"):
        if text.startswith(prefix):
            return text.removeprefix(prefix)
    return text


def load_stage_a_index(settings: SensitivitySettings) -> StageAIndex:
    """Every stage-A record on disk, indexed by what identifies it."""
    index: StageAIndex = {}
    for campaign in stage_a_campaigns(settings):
        window_label = str(campaign.window_label)
        kind = campaign_kind(campaign.tag, window_label)
        try:
            records = load_run_records(campaign.campaign_dir)
        except FileNotFoundError:
            continue
        for record in records:
            label = str((record.get("configuration") or {}).get("label", ""))
            index[(window_label, kind, label, str(record.get("method", "")))] = record
    return index


def _cell_records(
    index: StageAIndex,
    window: ScanWindow,
    configuration: SensitivityConfiguration,
) -> list[Mapping[str, Any]]:
    """The four runs one configuration selects, in method order."""
    label = configuration.scale.label
    wanted = (
        (configuration.reward_mode, configuration.methods[0]),
        (configuration.reward_mode, configuration.pmaddpg),
        (BASELINE_CAMPAIGN_SUFFIX, configuration.ltos),
        (BASELINE_CAMPAIGN_SUFFIX, configuration.mitm),
    )
    found = []
    for kind, method in wanted:
        record = index.get((window.label, kind, label, method))
        if record is not None:
            found.append(record)
    return found


def score_all_configurations(
    settings: SensitivitySettings, index: StageAIndex | None = None
) -> list[ConfigurationScore]:
    """Score every configuration of stage A (the environment is a coordinate)."""
    records = load_stage_a_index(settings) if index is None else index
    return [
        score_configuration(
            _cell_records(records, configuration.window, configuration),
            configuration,
            configuration.window,
        )
        for configuration in settings.configurations()
    ]


def select_stage_b_choices(
    scores: Sequence[ConfigurationScore], how_many: int
) -> list[StageBChoice]:
    """The top ``how_many`` configurations, each with the window it won on."""
    return [
        StageBChoice(rank=rank, configuration=score.configuration, window=score.window)
        for rank, score in enumerate(best_per_configuration(scores)[:how_many], start=1)
    ]


def _method_details(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for record in records:
        metrics = record.get("final_metrics") or {}
        absolute = record.get("final_metrics_absolute") or {}
        details[str(record.get("method", ""))] = {
            "metrics": {
                metric: float(metrics[metric])
                for metric in PUBLISHED_METRICS
                if metric in metrics
            },
            "absolute": {
                metric: float(absolute[metric])
                for metric in PUBLISHED_METRICS
                if metric in absolute
            },
            "total_conversions": record.get("total_conversions"),
            "saturation_iteration": record.get("saturation_iteration"),
            "runtime_seconds": record.get("runtime_seconds"),
        }
    return details


def stage_a_record(
    settings: SensitivitySettings,
    scores: Sequence[ConfigurationScore],
    index: StageAIndex,
) -> dict[str, Any]:
    """The whole screening stage, every configuration included."""
    best = best_per_configuration(scores)
    per_configuration = [
        {
            **score.as_record(),
            "methods": _method_details(
                _cell_records(index, score.window, score.configuration)
            ),
        }
        for score in best
    ]
    distribution = {str(count): 0 for count in range(len(PUBLISHED_METRICS) + 1)}
    for score in best:
        key = str(int(score.window_score.first_place_count))
        distribution[key] = distribution.get(key, 0) + 1
    return {
        "scenario": 1,
        "seed": settings.stage_a_seed,
        "iterations": settings.iterations,
        "weighted": False,
        "windows": [window.as_record() for window in settings.windows],
        "axes": {
            "environment": list(settings.environments),
            "cap": list(settings.caps),
            "qolu_reward_mode": list(settings.reward_modes),
            "pmaddpg": list(settings.pmaddpg_variants),
            "ltos": list(settings.ltos_variants),
            "mitm": list(settings.mitm_variants),
        },
        "n_configurations": len(settings.configurations()),
        "n_cells": len(scores),
        "closeness_view": PUBLISHED_CLOSENESS_VIEW,
        "configurations": per_configuration,
        "all_cells": [score.as_record() for score in scores],
        "score_distribution": distribution,
    }


def campaign_value_mae(summary: Mapping[str, Any]) -> dict[str, float | None]:
    """Mean absolute difference from **all 48** published cells, per view.

    Stage A compares one scenario's 12 cells; stages B and C run all four
    scenarios, so their closeness statistic is the mean over every published
    cell the campaign actually has.  Both views are returned, the headline one
    first, for the same reason stage A prints both: the published tables are on
    neither view, and printing only one would imply otherwise.
    """
    result: dict[str, float | None] = {}
    for view in ("absolute", "minmax", "raw"):
        differences: list[float] = []
        for configuration in summary.get("configurations", []):
            for block in configuration.get("scenarios", []):
                scenario = int(block["scenario"])
                values: dict[str, dict[str, float]] = {}
                for method, method_block in (block.get("methods") or {}).items():
                    for metric in PUBLISHED_METRICS:
                        entry = ((method_block or {}).get(view) or {}).get(metric)
                        if entry is None:
                            continue
                        values.setdefault(metric, {})[str(method)] = float(
                            entry["mean"]
                        )
                cell = value_closeness(values, scenario=scenario)
                if cell is not None:
                    differences.append(cell)
        result[view] = (
            float(sum(differences) / len(differences)) if differences else None
        )
    return result


def _campaign_entry(
    settings: SensitivitySettings,
    campaign: Any,
    choice: StageBChoice | None,
) -> dict[str, Any] | None:
    """Aggregate one campaign, write its documents and count its cells."""
    try:
        summary = build_summary(campaign.campaign_dir, metadata=campaign.metadata())
    except (FileNotFoundError, ValueError) as error:
        logger.warning("no records for %s (%s)", campaign.tag, error)
        return None
    records = load_run_records(campaign.campaign_dir)
    for path in write_campaign_documents(
        campaign.campaign_dir, summary, records=records, decimals=settings.decimals
    ).values():
        logger.info("wrote %s", path)
    alphabetical, total, alphabetical_cells = count_pattern_matches(summary)
    tie_aware, tie_total, tie_cells = count_pattern_matches_tie_aware(summary)
    return {
        "tag": campaign.tag,
        "rank": choice.rank if choice else None,
        "configuration": choice.configuration.as_record() if choice else None,
        "window": choice.window.as_record() if choice else None,
        "weighted": bool(campaign.weighted_consensus),
        "reading": "weighted" if campaign.weighted_consensus else "plain",
        "matched_cells_alphabetical": alphabetical,
        "matched_cells_tie_aware": tie_aware,
        "total_cells": max(total, tie_total),
        "value_mae": campaign_value_mae(summary),
        "closeness_view": PUBLISHED_CLOSENESS_VIEW,
        "cells_alphabetical": alphabetical_cells,
        "cells_tie_aware": tie_cells,
        "campaign_dir": str(campaign.campaign_dir),
        "n_runs": summary.get("n_runs", 0),
        "summary": summary,
    }


def stage_b_records(
    settings: SensitivitySettings, choices: Sequence[StageBChoice]
) -> list[dict[str, Any]]:
    """Aggregate each stage-B campaign and count its 12 cells, both ways."""
    entries: list[dict[str, Any]] = []
    for campaign in stage_b_campaigns(settings, choices):
        choice = next(
            (item for item in choices if campaign.tag.startswith(item.base_tag)), None
        )
        entry = _campaign_entry(settings, campaign, choice)
        if entry is not None:
            entries.append(entry)
    entries.sort(key=_stage_b_key, reverse=True)
    return entries


def _stage_b_key(entry: Mapping[str, Any]) -> tuple[int, int, float, str]:
    """Ordering matches first, then value closeness, then the tag for stability."""
    mae = (entry.get("value_mae") or {}).get(PUBLISHED_CLOSENESS_VIEW)
    return (
        int(entry.get("matched_cells_tie_aware", 0)),
        int(entry.get("matched_cells_alphabetical", 0)),
        float("-inf") if mae is None else -float(mae),
        str(entry.get("tag", "")),
    )


def stage_c_record(
    settings: SensitivitySettings,
    configuration: SensitivityConfiguration,
    weighted: bool,
) -> dict[str, Any] | None:
    """The deliverable campaign: the winning configuration at five seeds."""
    campaigns = stage_c_campaigns(settings, configuration, weighted)
    for campaign in campaigns:
        entry = _campaign_entry(settings, campaign, None)
        if entry is None:
            continue
        entry["configuration"] = configuration.as_record()
        entry["window"] = configuration.window.as_record()
        entry["seeds"] = list(settings.stage_c_seeds)
        return entry
    return None


def reference_records(
    settings: SensitivitySettings, index: StageAIndex
) -> list[dict[str, Any]]:
    """The documented reference runs - never swept, always shown."""
    from chapter4_qolu.window_scan import parse_window_label

    entries: list[dict[str, Any]] = []
    for reference in settings.reference_runs:
        window = parse_window_label(reference.environment)
        rows, cols = window.shape
        label = ScaleConfig(
            grid_size=rows,
            conversion_fraction=settings.conversion_fraction,
            max_conversions=reference.cap,
            grid_width=cols,
        ).label
        record = index.get(
            (window.label, f"{reference.method}", label, reference.method)
        )
        entries.append(
            {
                **reference.as_record(),
                "found": record is not None,
                "methods": _method_details([record] if record else []),
            }
        )
    return entries


def published_record() -> dict[str, Any]:
    """The published-normalisation finding, for the summary and the report."""
    return normalisation_check()


def stage_b2_records(
    settings: SensitivitySettings,
    configuration: SensitivityConfiguration,
    windows: Sequence[Any],
    weighted: bool = False,
) -> list[dict[str, Any]]:
    """Score every stage-B2 window under the winning recipe, best first."""
    from chapter4_qolu.sensitivity_b2 import score_b2_window
    from chapter4_qolu.sensitivity_stages import stage_b2_campaigns

    entries: list[dict[str, Any]] = []
    for campaign, window in zip(
        stage_b2_campaigns(settings, configuration, windows, weighted), windows
    ):
        try:
            records = load_run_records(campaign.campaign_dir)
        except FileNotFoundError:
            records = []
        score = score_b2_window(records, window, methods=configuration.methods)
        entries.append(
            {
                **score.as_record(),
                "tag": campaign.tag,
                "campaign_dir": str(campaign.campaign_dir),
                "methods": _method_details(records),
            }
        )
    entries.sort(key=lambda entry: entry["window"]["label"])
    entries.sort(
        key=lambda entry: (
            entry["score"]["first_place_count"],
            entry["score"]["kendall_sum"],
            -(entry["value_mae"] if entry["value_mae"] is not None else 1e9),
        ),
        reverse=True,
    )
    return entries


def stage_b2_windows_on_disk(settings: SensitivitySettings) -> list[Any]:
    """The stage-B2 windows that were actually run, read from the directories.

    Reporting must not need the raster: ``--report-only`` is meant to rebuild
    every document from the records alone, and the stage-B2 window list would
    otherwise have to be re-derived from the scan grid (which needs the raster
    loaded, and which could silently disagree with what was run).  Each
    campaign directory is named ``campaign_b2_<window label>``, so the list of
    windows *is* the list of directories.
    """
    from chapter4_qolu.sensitivity_plan import sensitivity_paths
    from chapter4_qolu.window_scan import parse_window_label

    root = sensitivity_paths(settings).stage_b2
    if not root.is_dir():
        return []
    windows = []
    for path in sorted(root.glob("campaign_b2_*")):
        label = path.name.removeprefix("campaign_b2_")
        try:
            windows.append(parse_window_label(label))
        except ValueError:
            logger.warning("stage B2: cannot parse window label %r", label)
    return windows
