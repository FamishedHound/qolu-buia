"""How the sweep's three stages are laid out as ordinary campaigns.

Split out of :mod:`chapter4_qolu.sensitivity_plan` so that *what* the sweep
searches stays separate from *how* those runs are arranged on disk - the same
split :mod:`chapter4_qolu.scan_plan` makes for the window scan, and for the
same reason: every stage is expressed as ordinary
:class:`~chapter4_qolu.campaign_plan.CampaignSettings`, so the result layout,
``--resume`` and the five campaign documents are the ordinary campaign
machinery.

============  =====================================================
stage         campaigns
============  =====================================================
A             one per (environment, reward reading), plus one for
              the reward-free baselines and one per reference run
B             two per confirmed configuration - plain and weighted
C             one, the winner, at five seeds
============  =====================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from chapter4_qolu.campaign_plan import CampaignSettings, campaign_runs
from chapter4_qolu.methods import QOLU_METHOD
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.scenario_worker import RunSpec
from chapter4_qolu.sensitivity_plan import (
    BASELINE_CAMPAIGN_SUFFIX,
    REWARD_MODES,
    SensitivityConfiguration,
    SensitivitySettings,
    StageBChoice,
    sensitivity_paths,
)
from chapter4_qolu.window_scan import ScanWindow, parse_window_label

__all__ = [
    "stage_a_campaigns",
    "stage_a_runs",
    "stage_a_tag",
    "stage_b2_campaigns",
    "stage_b2_runs",
    "stage_b2_tag",
    "stage_b_campaigns",
    "stage_b_runs",
    "stage_c_campaigns",
    "stage_c_runs",
]


def _campaign(
    settings: SensitivitySettings,
    window: ScanWindow,
    tag: str,
    output_dir: Path,
    methods: Sequence[str],
    scenarios: Sequence[int],
    seeds: Sequence[int],
    reward_mode: str,
    weighted: bool,
    scales: Sequence[ScaleConfig],
) -> CampaignSettings:
    easting, northing = window.bng_top_left_km
    return CampaignSettings(
        raster=settings.raster,
        tag=tag,
        methods=tuple(methods),
        scenarios=tuple(scenarios),
        seeds=tuple(seeds),
        configs=tuple(scales),
        iterations=settings.iterations,
        stakeholders=settings.stakeholders,
        qolu_reward_mode=reward_mode,
        scale_by_scenario_weight=weighted,
        weighted_consensus=weighted,
        max_distance_km=settings.max_distance_km,
        parallel=settings.parallel,
        resume=settings.resume,
        window_bng_km=(float(easting), float(northing)),
        window_label=window.label,
        region_mask=settings.region_mask,
        decimals=settings.decimals,
        output_dir=output_dir,
    )


def stage_a_tag(window: ScanWindow, reward_mode: str) -> str:
    """Campaign tag of one stage-A screening campaign."""
    return f"a_{window.label}_{reward_mode}"


def stage_a_campaigns(settings: SensitivitySettings) -> list[CampaignSettings]:
    """One campaign per (window, reward mode), plus one for the baselines.

    Design note (why the baselines get their own campaign): LToS and MitM are
    never handed a ``QOLURewardConfig`` - ``scenario_worker._build_baseline``
    passes one only to P-MADDPG - so running them once per reward mode would
    triple the cheap runs and produce three byte-identical records.  They are
    therefore run once per (window, cap) under the default ``per_parcel``
    label, and every configuration reuses them whatever its reward reading.
    """
    paths = sensitivity_paths(settings)
    campaigns: list[CampaignSettings] = []
    for window in settings.windows:
        scales = settings.scales_for(window)
        for mode in settings.reward_modes:
            methods = (QOLU_METHOD, *settings.pmaddpg_variants)
            campaigns.append(
                _campaign(
                    settings,
                    window,
                    tag=stage_a_tag(window, mode),
                    output_dir=paths.stage_a,
                    methods=methods,
                    scenarios=(1,),
                    seeds=(settings.stage_a_seed,),
                    reward_mode=mode,
                    weighted=False,
                    scales=scales,
                )
            )
        baseline_methods = (*settings.ltos_variants, *settings.mitm_variants)
        campaigns.append(
            _campaign(
                settings,
                window,
                tag=stage_a_tag(window, BASELINE_CAMPAIGN_SUFFIX),
                output_dir=paths.stage_a,
                methods=baseline_methods,
                scenarios=(1,),
                seeds=(settings.stage_a_seed,),
                reward_mode=REWARD_MODES[0],
                weighted=False,
                scales=scales,
            )
        )
    campaigns.extend(_reference_campaigns(settings))
    return campaigns


def _reference_campaigns(settings: SensitivitySettings) -> list[CampaignSettings]:
    """One tiny campaign per documented reference run (never swept)."""
    paths = sensitivity_paths(settings)
    campaigns: list[CampaignSettings] = []
    for reference in settings.reference_runs:
        window = parse_window_label(reference.environment)
        rows, cols = window.shape
        campaigns.append(
            _campaign(
                settings,
                window,
                tag=f"ref_{window.label}_{reference.method}",
                output_dir=paths.stage_a,
                methods=(reference.method,),
                scenarios=(1,),
                seeds=(settings.stage_a_seed,),
                reward_mode=reference.reward_mode,
                weighted=False,
                scales=(
                    ScaleConfig(
                        grid_size=rows,
                        conversion_fraction=settings.conversion_fraction,
                        max_conversions=reference.cap,
                        grid_width=cols,
                    ),
                ),
            )
        )
    return campaigns


def stage_a_runs(settings: SensitivitySettings) -> tuple[RunSpec, ...]:
    """Every stage-A run, de-duplicated by output path."""
    specs: dict[Path, RunSpec] = {}
    for campaign in stage_a_campaigns(settings):
        for spec in campaign_runs(campaign):
            specs.setdefault(spec.output_path, spec)
    return tuple(specs.values())


def stage_b_campaigns(
    settings: SensitivitySettings, choices: Sequence[StageBChoice]
) -> list[CampaignSettings]:
    """Two campaigns per choice: the plain and the weighted consensus reading."""
    paths = sensitivity_paths(settings)
    campaigns: list[CampaignSettings] = []
    for choice in choices:
        for weighted, suffix in ((False, "plain"), (True, "weighted")):
            campaigns.append(
                _campaign(
                    settings,
                    choice.window,
                    tag=f"{choice.base_tag}_{suffix}",
                    output_dir=paths.stage_b,
                    methods=choice.configuration.methods,
                    scenarios=settings.stage_b_scenarios,
                    seeds=settings.stage_b_seeds,
                    reward_mode=choice.configuration.reward_mode,
                    weighted=weighted,
                    scales=(choice.configuration.scale,),
                )
            )
    return campaigns


def stage_b_runs(
    settings: SensitivitySettings, choices: Sequence[StageBChoice]
) -> tuple[RunSpec, ...]:
    specs: dict[Path, RunSpec] = {}
    for campaign in stage_b_campaigns(settings, choices):
        for spec in campaign_runs(campaign):
            specs.setdefault(spec.output_path, spec)
    return tuple(specs.values())


def stage_c_campaigns(
    settings: SensitivitySettings,
    configuration: SensitivityConfiguration,
    weighted: bool,
) -> list[CampaignSettings]:
    """The deliverable: one campaign, the winning configuration, five seeds."""
    paths = sensitivity_paths(settings)
    suffix = "weighted" if weighted else "plain"
    return [
        _campaign(
            settings,
            configuration.window,
            tag=f"c_{configuration.label}_{suffix}",
            output_dir=paths.stage_c,
            methods=configuration.methods,
            scenarios=settings.stage_b_scenarios,
            seeds=settings.stage_c_seeds,
            reward_mode=configuration.reward_mode,
            weighted=weighted,
            scales=(configuration.scale,),
        )
    ]


def stage_c_runs(
    settings: SensitivitySettings,
    configuration: SensitivityConfiguration,
    weighted: bool,
) -> tuple[RunSpec, ...]:
    specs: dict[Path, RunSpec] = {}
    for campaign in stage_c_campaigns(settings, configuration, weighted):
        for spec in campaign_runs(campaign):
            specs.setdefault(spec.output_path, spec)
    return tuple(specs.values())


def stage_b2_tag(window: ScanWindow) -> str:
    """Campaign tag of one stage-B2 window."""
    return f"b2_{window.label}"


def stage_b2_campaigns(
    settings: SensitivitySettings,
    configuration: SensitivityConfiguration,
    windows: Sequence[ScanWindow],
    weighted: bool = False,
) -> list[CampaignSettings]:
    """The winning recipe on every candidate window, Scenario 1, seed 0.

    Only the *window* changes: the cap, the QOLU reward reading, the consensus
    reading and the three baseline variants are the winner's, which is what
    makes the stage a test of the window rather than of the recipe.  The cap is
    re-expressed at each window's own extent, so the whole-region box runs at
    210 x 240 and the scan windows at 50 x 50.
    """
    paths = sensitivity_paths(settings)
    campaigns: list[CampaignSettings] = []
    for window in windows:
        rows, cols = window.shape
        campaigns.append(
            _campaign(
                settings,
                window,
                tag=stage_b2_tag(window),
                output_dir=paths.stage_b2,
                methods=configuration.methods,
                scenarios=(1,),
                seeds=(settings.stage_a_seed,),
                reward_mode=configuration.reward_mode,
                weighted=weighted,
                scales=(
                    ScaleConfig(
                        grid_size=rows,
                        conversion_fraction=settings.conversion_fraction,
                        max_conversions=configuration.cap,
                        grid_width=cols,
                    ),
                ),
            )
        )
    return campaigns


def stage_b2_runs(
    settings: SensitivitySettings,
    configuration: SensitivityConfiguration,
    windows: Sequence[ScanWindow],
    weighted: bool = False,
) -> tuple[RunSpec, ...]:
    specs: dict[Path, RunSpec] = {}
    for campaign in stage_b2_campaigns(settings, configuration, windows, weighted):
        for spec in campaign_runs(campaign):
            specs.setdefault(spec.output_path, spec)
    return tuple(specs.values())
