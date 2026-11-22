"""What the Southeast window scan consists of, and what it will cost.

Split out of :mod:`chapter4_qolu.run_window_scan` so that "which runs" stays
separate from "how they are executed" - the same split
:mod:`chapter4_qolu.campaign_plan` makes for an ordinary campaign, and for the
same reason: nothing here loads a raster, builds an environment or imports
torch, so the plan and the ETA can be inspected instantly.

Each stage of the scan is expressed as ordinary
:class:`~chapter4_qolu.campaign_plan.CampaignSettings`, one per window, with the
window pinned by the British National Grid coordinates of its top-left corner.
Everything downstream - the result layout, ``--resume``, the four campaign
documents - is therefore the ordinary campaign machinery.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from chapter4_qolu.campaign_plan import (
    DEFAULT_METHODS,
    CampaignSettings,
    campaign_runs,
    estimate_runtime_seconds,
)
from chapter4_qolu.experiment import DEFAULT_RESULTS_DIR
from chapter4_qolu.reporting.paths import ScaleConfig, parse_scale_config
from chapter4_qolu.window_scan import (
    MAX_SEA_FRACTION,
    SOUTH_EAST_ENGLAND,
    WINDOW_SIZE_KM,
    WINDOW_STRIDE_KM,
    BoundingBoxKm,
    ScanWindow,
)

__all__ = [
    "PUBLISHED_CONFIG",
    "SINGLE_THREAD_COST_FACTOR",
    "ScanPaths",
    "ScanSettings",
    "estimate_stage_seconds",
    "scan_paths",
    "stage1_settings",
    "stage2_settings",
    "stage_campaigns",
]


#: The published scale: 50 x 50 parcels, whole-parcel conversion, cap 10.
PUBLISHED_CONFIG: ScaleConfig = parse_scale_config("50:1.0:10")

#: Design note: a worker pinned to one torch thread costs this many times what
#: ``campaign_plan.METHOD_COST_SECONDS`` says.  Measured on this machine at the
#: published scale on the GB dominant-class raster, 50 iterations per method,
#: ``torch_threads = 1``: QOLU + BUIA 337 s / 1000 iterations against the
#: 120 s of the multi-threaded table, P-MADDPG 275 s against 110 s, LToS 46 s
#: against 45 s, MitM 30 s against 25 s.  The two neural methods dominate the
#: total, so 2.3 is the ratio of the *sums*, not of any single method.
SINGLE_THREAD_COST_FACTOR: float = 2.3


@dataclass(frozen=True)
class ScanSettings:
    """What the scan searches and how hard."""

    raster: Path
    tag: str = "se_20260924"
    box: BoundingBoxKm = SOUTH_EAST_ENGLAND
    window_size_km: int = WINDOW_SIZE_KM
    stride_km: int = WINDOW_STRIDE_KM
    max_sea_fraction: float = MAX_SEA_FRACTION
    methods: tuple[str, ...] = DEFAULT_METHODS
    iterations: int = 1000
    config: ScaleConfig = field(default=PUBLISHED_CONFIG)
    qolu_reward_mode: str = "per_parcel"
    stakeholders: int = 3
    #: Design note: three windows go through to stage 2.  Enough to show whether
    #: the stage-1 winner is a fluke of one landscape, few enough that the
    #: confirmation stage costs about as much as the screening one.
    stage2_windows: int = 3
    stage2_seeds: tuple[int, ...] = (0, 1, 2)
    max_distance_km: float = 10.0
    decimals: int = 3
    parallel: int = 16
    resume: bool = False
    output_dir: Path = DEFAULT_RESULTS_DIR

    def __post_init__(self) -> None:
        object.__setattr__(self, "raster", Path(self.raster))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "methods", tuple(self.methods))
        object.__setattr__(
            self, "stage2_seeds", tuple(int(s) for s in self.stage2_seeds)
        )
        if not self.stage2_seeds:
            raise ValueError("at least one stage-2 seed is required")
        if self.stage2_windows < 1:
            raise ValueError("stage2_windows must be positive")
        if not 0.0 <= self.max_sea_fraction <= 1.0:
            raise ValueError("max_sea_fraction must lie in [0, 1]")


@dataclass(frozen=True)
class ScanPaths:
    """Where a scan writes."""

    root: Path
    stage1: Path
    stage2: Path
    summary: Path
    report: Path


def scan_paths(settings: ScanSettings) -> ScanPaths:
    """The output layout of one scan."""
    root = Path(settings.output_dir) / f"window_scan_{settings.tag}"
    return ScanPaths(
        root=root,
        stage1=root / "stage1",
        stage2=root / "stage2",
        summary=root / "scan_summary.json",
        report=root / "scan.md",
    )


def _campaign(
    settings: ScanSettings,
    window: ScanWindow,
    output_dir: Path,
    scenarios: tuple[int, ...],
    seeds: tuple[int, ...],
    weighted: bool,
) -> CampaignSettings:
    easting, northing = window.bng_top_left_km
    return CampaignSettings(
        raster=settings.raster,
        tag=window.label,
        methods=settings.methods,
        scenarios=scenarios,
        seeds=seeds,
        configs=(settings.config,),
        iterations=settings.iterations,
        stakeholders=settings.stakeholders,
        qolu_reward_mode=settings.qolu_reward_mode,
        scale_by_scenario_weight=weighted,
        weighted_consensus=weighted,
        max_distance_km=settings.max_distance_km,
        parallel=settings.parallel,
        resume=settings.resume,
        window_bng_km=(float(easting), float(northing)),
        window_label=window.label,
        decimals=settings.decimals,
        output_dir=output_dir,
    )


def stage1_settings(settings: ScanSettings, window: ScanWindow) -> CampaignSettings:
    """Screening: Scenario 1, seed 0, the literal (unweighted) reading."""
    return _campaign(
        settings,
        window,
        scan_paths(settings).stage1,
        scenarios=(1,),
        seeds=(0,),
        weighted=False,
    )


def stage2_settings(settings: ScanSettings, window: ScanWindow) -> CampaignSettings:
    """Confirmation: four scenarios, several seeds, the weighted reading."""
    return _campaign(
        settings,
        window,
        scan_paths(settings).stage2,
        scenarios=(1, 2, 3, 4),
        seeds=settings.stage2_seeds,
        weighted=True,
    )


def stage_campaigns(
    settings: ScanSettings, windows: Sequence[ScanWindow], stage: int
) -> list[CampaignSettings]:
    builder = stage1_settings if stage == 1 else stage2_settings
    return [builder(settings, window) for window in windows]


def estimate_stage_seconds(
    settings: ScanSettings, windows: Sequence[ScanWindow], stage: int
) -> float:
    """Estimated *serial* CPU cost of one stage, in seconds.

    :data:`~chapter4_qolu.campaign_plan.METHOD_COST_SECONDS` was measured with
    torch free to use every core, and its docstring says so; a scan worker is
    pinned to one thread, which the scan corrects for with
    :data:`SINGLE_THREAD_COST_FACTOR` so that the printed ETA is a usable
    number rather than a documented under-estimate.  The campaign estimator
    itself does not apply the factor.
    """
    serial = sum(
        estimate_runtime_seconds(campaign_runs(campaign))
        for campaign in stage_campaigns(settings, windows, stage)
    )
    return serial * SINGLE_THREAD_COST_FACTOR
