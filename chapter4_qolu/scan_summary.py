"""Turning the scan's run records into ``scan_summary.json``.

Every piece here reads what is on disk and returns plain JSON-ready data: the
grid that was searched, each window's composition and stage-1 score, the
distribution of those scores, and - for the windows that went through to stage
2 - the ordinary campaign summary plus the count of (scenario, metric) cells
that reproduce the published ordering.

Kept apart from :mod:`chapter4_qolu.run_window_scan` so the summary (and with
it ``scan.md``) can be rebuilt from the records alone, without running or even
loading a single environment.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from chapter4_qolu.data.lcm_loader import LCMRaster
from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS, load_run_records
from chapter4_qolu.reporting.documents import build_summary, write_campaign_documents
from chapter4_qolu.scan_plan import ScanSettings, stage1_settings, stage2_settings
from chapter4_qolu.scan_scoring import count_pattern_matches, score_window
from chapter4_qolu.window_scan import ScanWindow, WindowComposition, window_origins

__all__ = [
    "grid_record",
    "metric_values_from_records",
    "raster_record",
    "score_distribution",
    "stage1_window_records",
    "stage2_window_records",
]

logger = logging.getLogger(__name__)


def metric_values_from_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Pivot per-run records into ``values[metric][method]`` (RAW view)."""
    values: dict[str, dict[str, float]] = {}
    for record in records:
        method = str(record.get("method", ""))
        metrics = record.get("final_metrics") or {}
        for metric in PUBLISHED_METRICS:
            if metric in metrics:
                values.setdefault(metric, {})[method] = float(metrics[metric])
    return values


def _method_details(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-method metrics, saturation iteration and conversion count."""
    details: dict[str, Any] = {}
    for record in records:
        metrics = record.get("final_metrics") or {}
        details[str(record.get("method", ""))] = {
            "metrics": {
                metric: float(metrics[metric])
                for metric in PUBLISHED_METRICS
                if metric in metrics
            },
            "total_conversions": record.get("total_conversions"),
            "saturation_iteration": record.get("saturation_iteration"),
            "runtime_seconds": record.get("runtime_seconds"),
        }
    return details


def stage1_window_records(
    settings: ScanSettings,
    compositions: Sequence[WindowComposition],
) -> list[dict[str, Any]]:
    """Score every retained window from the stage-1 records on disk."""
    scored: list[dict[str, Any]] = []
    for composition in compositions:
        campaign = stage1_settings(settings, composition.window)
        try:
            records = load_run_records(campaign.campaign_dir)
        except FileNotFoundError:
            records = []
        score = score_window(metric_values_from_records(records))
        scored.append(
            {
                "window": composition.window.as_record(),
                "composition": composition.as_record(),
                "score": score.as_record(),
                "methods": _method_details(records),
                "n_records": len(records),
            }
        )
    scored.sort(
        key=lambda entry: (
            entry["score"]["first_place_count"],
            entry["score"]["kendall_sum"],
            entry["score"]["reference_minmax_mean"],
            # A stable, readable final tie-break so a rerun cannot reshuffle
            # equally-scoring windows.
            entry["window"]["label"],
        ),
        reverse=True,
    )
    return scored


def score_distribution(scored: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    distribution = {str(count): 0 for count in range(len(PUBLISHED_METRICS) + 1)}
    for entry in scored:
        key = str(int(entry["score"]["first_place_count"]))
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def raster_record(raster: LCMRaster) -> dict[str, Any]:
    return {
        "path": str(raster.path),
        "name": raster.path.name,
        "crs": raster.crs,
        "pixel_size_m": raster.pixel_size_m,
        "height": raster.height,
        "width": raster.width,
        "origin_xy_m": list(raster.origin_xy) if raster.origin_xy else None,
    }


def grid_record(
    settings: ScanSettings, n_candidates: int, n_retained: int
) -> dict[str, Any]:
    box = settings.box
    return {
        "bounding_box_km": box.as_record(),
        "bounding_box": box.description,
        "window_size_km": settings.window_size_km,
        "stride_km": settings.stride_km,
        "max_sea_fraction": settings.max_sea_fraction,
        "eastings": list(
            window_origins(
                box.east_min, box.east_max, settings.window_size_km, settings.stride_km
            )
        ),
        "northings": list(
            window_origins(
                box.north_min,
                box.north_max,
                settings.window_size_km,
                settings.stride_km,
            )
        ),
        "n_candidates": n_candidates,
        "n_retained": n_retained,
    }


def stage2_window_records(
    settings: ScanSettings, windows: Sequence[ScanWindow]
) -> list[dict[str, Any]]:
    """Aggregate each stage-2 window into its campaign documents and count cells."""
    entries: list[dict[str, Any]] = []
    for window in windows:
        campaign = stage2_settings(settings, window)
        try:
            summary = build_summary(campaign.campaign_dir, metadata=campaign.metadata())
        except (FileNotFoundError, ValueError) as error:
            logger.warning("stage 2: no records for %s (%s)", window.label, error)
            continue
        records = load_run_records(campaign.campaign_dir)
        for path in write_campaign_documents(
            campaign.campaign_dir, summary, records=records, decimals=settings.decimals
        ).values():
            logger.info("wrote %s", path)
        matched, total, cells = count_pattern_matches(summary)
        entries.append(
            {
                "window": window.as_record(),
                "matched_cells": matched,
                "total_cells": total,
                "cells": cells,
                "campaign_dir": str(campaign.campaign_dir),
                "summary": summary,
            }
        )
    entries.sort(
        key=lambda entry: (entry["matched_cells"], entry["window"]["label"]),
        reverse=True,
    )
    return entries
