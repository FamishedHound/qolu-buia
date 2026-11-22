"""What a four-scenario campaign consists of, and what it will cost.

Split out of :mod:`chapter4_qolu.run_scenarios` so that "which runs" stays
separate from "how they are executed" (:mod:`chapter4_qolu.scenario_worker`)
and from the driver itself.  Nothing here loads a raster, builds an environment
or imports torch, so the plan and the ETA can be inspected instantly::

    python -m chapter4_qolu.run_scenarios --raster ... --dry-run
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from chapter4_qolu.config import SCENARIOS, QOLURewardConfig
from chapter4_qolu.experiment import DEFAULT_RESULTS_DIR
from chapter4_qolu.reporting.documents import DEFAULT_DECIMALS
from chapter4_qolu.reporting.paths import (
    ScaleConfig,
    campaign_directory,
    parse_scale_config,
    run_json_path,
    should_run,
)
from chapter4_qolu.scenario_worker import METHOD_KEYS, QOLU_METHOD, RunSpec

__all__ = [
    "DEFAULT_CONFIGS",
    "DEFAULT_METHODS",
    "DEFAULT_SCENARIOS",
    "DEFAULT_SEEDS",
    "METHOD_COST_SECONDS",
    "CampaignSettings",
    "campaign_runs",
    "estimate_runtime_seconds",
    "format_duration",
    "pending_runs",
    "smoke_settings",
]

#: The four methods the paper's tables compare.
DEFAULT_METHODS: tuple[str, ...] = (QOLU_METHOD, "pmaddpg", "ltos", "mitm")
DEFAULT_SCENARIOS: tuple[int, ...] = (1, 2, 3, 4)
#: Design note: five seeds.  The paper reports standard errors but never states
#: how many runs they come from; five is the smallest count that makes a paired
#: t-test (four degrees of freedom) worth quoting.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
#: ``50:1.0:10`` is the published scale; ``100:0.25:10`` is the non-saturating
#: configuration found by ``sweep_saturation`` (the published one urbanises the
#: whole 50 x 50 map after ~340-430 of the 1000 iterations, so the rest of the
#: run is a no-op).
DEFAULT_CONFIGS: tuple[str, ...] = ("50:1.0:10", "100:0.25:10")

#: Measured CPU cost of 1000 iterations on a 50 x 50 crop, seconds per method,
#: with torch free to use every core; a worker pinned to one thread is slower,
#: so an ETA built from this table is a lower bound (the window scan and the
#: sensitivity sweep correct for it with
#: :data:`chapter4_qolu.scan_plan.SINGLE_THREAD_COST_FACTOR`).
#:
#: Design note: ``fixed_ltos`` costs what ``ltos`` does (it only skips the
#: high-level update) and ``mitm_learned`` what ``mitm`` does (it only adds a
#: tabular nudge).  ``pmaddpg`` does not: its published
#: settings add epsilon-greedy exploration over every candidate and a hard
#: target update, and a full 1000-iteration run on the 210 x 240 region
#: measured 1,170-1,224 s against QOLU + BUIA's 539-563 s, so it is priced at
#: roughly 2.2x QOLU + BUIA.  ``pmaddpg_v1`` and ``pmaddpg_reg_only`` keep the
#: lighter settings and their measured 110 s.  (A *cold*
#: 20-iteration run looks five times dearer still; that is torch's one-off
#: warm-up, which a 1000-iteration run amortises, and it must not be
#: extrapolated from.)
#:
#: The QOLU variants are priced from a measured micro-benchmark
#: at the campaign's own settings - 9 x 9 observation window, 11 channels,
#: mini-batch 256, one torch thread, 20 act + learn iterations:
#:
#: ====================================  =========  =============
#: variant                               parameters  ms/iteration
#: ====================================  =========  =============
#: ``qolu_c51`` (plain linear layers)    2,920,921          222.8
#: ``qolu_buia`` (noisy, ``all_linear``) 5,746,290          283.3
#: (noisy, ``streams`` scope)            2,960,242          233.5
#: ``qolu_buia_per``                     5,746,290          281.4
#: ====================================  =========  =============
#:
#: So the NoisyNet layers of Algorithm 1 line 5 cost ~27% under the default
#: ``all_linear`` scope (they double the width of every fully-connected layer's
#: parameter tensor, and the projection out of the flattened convolution stack
#: is the expensive one), ~5% under ``streams``; prioritised replay is free
#: within noise.  ``qolu_buia`` is therefore priced at 1.27x the 120 s
#: ``qolu_c51`` costs.
METHOD_COST_SECONDS: dict[str, float] = {
    QOLU_METHOD: 152.0,
    "qolu_buia_per": 152.0,
    "qolu_c51": 120.0,
    "pmaddpg": 260.0,
    "pmaddpg_v1": 110.0,
    "pmaddpg_reg_only": 110.0,
    "ltos": 45.0,
    "fixed_ltos": 45.0,
    "mitm": 25.0,
    "mitm_learned": 25.0,
    "no_regulator": 10.0,
}
_COST_ITERATIONS = 1000
_COST_GRID_SIZE = 50
#: Design note: cost grows linearly in the number of parcels between the two
#: measured extents - 100 x 100 was 25% dearer than 50 x 50, so the factor is
#: ``1 + 0.25 * (cells / 2500 - 1) / 3``.  Most of the per-iteration work is the
#: fixed 50-candidate decision buffer, which is why the growth is so mild.
_GRID_COST_AT_100 = 1.25


@dataclass(frozen=True)
class CampaignSettings:
    """Which runs a campaign performs and where they are written."""

    raster: Path
    tag: str = "main"
    methods: tuple[str, ...] = DEFAULT_METHODS
    scenarios: tuple[int, ...] = DEFAULT_SCENARIOS
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    configs: tuple[ScaleConfig, ...] = ()
    iterations: int = 1000
    stakeholders: int = 3
    qolu_reward_mode: str = "per_parcel"
    #: Optional routes by which the scenario weights reach the top-down agents
    #: (QOLU + BUIA and P-MADDPG); see
    #: :class:`~chapter4_qolu.config.QOLURewardConfig`.  Design note: both stay
    #: off by default, because turning either on changes results and the
    #: published reading is the unweighted one.  A campaign that turns them on
    #: is kept apart by its ``--tag``; the directory layout is unchanged.
    scale_by_scenario_weight: bool = False
    weighted_consensus: bool = False
    #: Physical distance at which the proximity metrics saturate in view (b).
    #: Design note: 10 km.  The paper never states one - its metrics are simply
    #: "normalized between 0 and 1" - and ``RewardConfig.max_distance = None``
    #: falls back to the grid diagonal, which makes the RAW ``land_use`` column
    #: depend on the extent rather than on the landscape.  10 km is the round
    #: number closest to the distance beyond which "far from bog, mountain and
    #: heath" stops being a meaningful planning distinction at 1 km parcels.
    max_distance_km: float = 10.0
    #: Design note: 16 workers by default (the machine this was built on has 32
    #: logical CPUs); each worker is pinned to one torch thread.
    parallel: int = 16
    resume: bool = False
    #: Optional pinned raster window shared by every run of the campaign - the
    #: Southeast window scan pins one per window, an ordinary campaign leaves
    #: both ``None`` and keeps the automatic densest-land crop.
    window_origin: tuple[int, int] | None = None
    window_bng_km: tuple[float, float] | None = None
    window_label: str | None = None
    #: Optional polygon boundary clipping the window; see
    #: :mod:`chapter4_qolu.data.region_mask`.  ``None`` = the bounding box.
    region_mask: Path | None = None
    #: Decimal places in ``tables.md`` / ``tables.tex``.  ``summary.json`` and
    #: ``per_seed.csv`` always hold full double precision.
    decimals: int = DEFAULT_DECIMALS
    output_dir: Path = DEFAULT_RESULTS_DIR

    def __post_init__(self) -> None:
        if not self.configs:
            object.__setattr__(
                self,
                "configs",
                tuple(parse_scale_config(text) for text in DEFAULT_CONFIGS),
            )
        object.__setattr__(self, "methods", tuple(self.methods))
        object.__setattr__(self, "scenarios", tuple(int(k) for k in self.scenarios))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        if not self.methods:
            raise ValueError("at least one method is required")
        unknown = [method for method in self.methods if method not in METHOD_KEYS]
        if unknown:
            raise ValueError(f"unknown method(s) {unknown}; expected {METHOD_KEYS}")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be distinct")
        if not self.scenarios:
            raise ValueError("at least one scenario is required")
        unknown_scenarios = [k for k in self.scenarios if k not in SCENARIOS]
        if unknown_scenarios:
            raise ValueError(f"unknown scenario(s) {unknown_scenarios}; expected 1-4")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.stakeholders < 1:
            raise ValueError("at least one stakeholder is required")
        if self.max_distance_km <= 0.0:
            raise ValueError("max_distance_km must be positive")
        if self.parallel < 1:
            raise ValueError("parallel must be at least one")
        if self.decimals < 1:
            raise ValueError("decimals must be positive")
        if self.window_origin is not None:
            object.__setattr__(
                self, "window_origin", tuple(int(value) for value in self.window_origin)
            )
        if self.window_bng_km is not None:
            object.__setattr__(
                self,
                "window_bng_km",
                tuple(float(value) for value in self.window_bng_km),
            )
        QOLURewardConfig(
            mode=self.qolu_reward_mode,
            scale_by_scenario_weight=self.scale_by_scenario_weight,
            weighted_consensus=self.weighted_consensus,
        )  # validates the reward reading

    @property
    def campaign_dir(self) -> Path:
        """Folder holding every record and document of this campaign."""
        return campaign_directory(self.output_dir, self.tag)

    def with_resume(self, resume: bool = True) -> CampaignSettings:
        """A copy that skips runs already on disk."""
        return replace(self, resume=resume)

    def metadata(self) -> dict[str, Any]:
        """JSON-serialisable description stored in ``summary.json``."""
        now = dt.datetime.now(dt.timezone.utc)
        return {
            "tag": self.tag,
            "generated": now.isoformat(timespec="seconds"),
            "raster": Path(self.raster).name,
            "iterations": self.iterations,
            "methods": list(self.methods),
            "scenarios": list(self.scenarios),
            "seeds": list(self.seeds),
            "configurations": [scale.label for scale in self.configs],
            "stakeholders": self.stakeholders,
            "qolu_reward_mode": self.qolu_reward_mode,
            "scale_by_scenario_weight": bool(self.scale_by_scenario_weight),
            "weighted_consensus": bool(self.weighted_consensus),
            "max_distance_km": self.max_distance_km,
            "window_label": self.window_label,
            "window_bng_top_left_km": (
                list(self.window_bng_km) if self.window_bng_km is not None else None
            ),
            "window_raster_origin": (
                list(self.window_origin) if self.window_origin is not None else None
            ),
            "region_mask": (
                Path(self.region_mask).name if self.region_mask is not None else None
            ),
            "table_decimals": self.decimals,
        }


def campaign_runs(settings: CampaignSettings) -> tuple[RunSpec, ...]:
    """Every run of the campaign, in configuration / scenario / seed order."""
    directory = settings.campaign_dir
    return tuple(
        RunSpec(
            raster=Path(settings.raster),
            method=method,
            scenario=scenario,
            seed=seed,
            scale=scale,
            output_path=run_json_path(directory, scale, scenario, method, seed),
            iterations=settings.iterations,
            stakeholders=settings.stakeholders,
            qolu_reward_mode=settings.qolu_reward_mode,
            scale_by_scenario_weight=settings.scale_by_scenario_weight,
            weighted_consensus=settings.weighted_consensus,
            max_distance_km=settings.max_distance_km,
            window_origin=settings.window_origin,
            window_bng_km=settings.window_bng_km,
            window_label=settings.window_label,
            region_mask=settings.region_mask,
        )
        for scale in settings.configs
        for scenario in settings.scenarios
        for seed in settings.seeds
        for method in settings.methods
    )


def pending_runs(
    specs: Sequence[RunSpec], resume: bool
) -> tuple[tuple[RunSpec, ...], tuple[RunSpec, ...]]:
    """Split runs into "still to do" and "already on disk"."""
    todo = tuple(spec for spec in specs if should_run(spec.output_path, resume))
    done = tuple(spec for spec in specs if not should_run(spec.output_path, resume))
    return todo, done


def _grid_cost_factor(n_cells: int) -> float:
    """Cost of one run of ``n_cells`` parcels, relative to the 50 x 50 figures.

    Expressed in *cells* rather than in a side length so that a rectangular
    extent is priced too; for a square grid it equals the side-length form.
    """
    cells = float(n_cells)
    reference = float(_COST_GRID_SIZE * _COST_GRID_SIZE)
    growth = (cells / reference - 1.0) / (100.0 * 100.0 / reference - 1.0)
    return 1.0 + (_GRID_COST_AT_100 - 1.0) * growth


def estimate_runtime_seconds(specs: Sequence[RunSpec]) -> float:
    """Estimated *serial* CPU cost of a set of runs, in seconds."""
    total = 0.0
    for spec in specs:
        base = METHOD_COST_SECONDS.get(spec.method, METHOD_COST_SECONDS[QOLU_METHOD])
        total += (
            base
            * (spec.iterations / _COST_ITERATIONS)
            * _grid_cost_factor(spec.scale.n_cells)
        )
    return total


def format_duration(seconds: float) -> str:
    """Human-readable duration: seconds, minutes or hours."""
    value = float(seconds)
    if value < 90.0:
        return f"{value:.0f} s"
    if value < 5400.0:
        return f"{value / 60.0:.1f} min"
    return f"{value / 3600.0:.1f} h"


def smoke_settings(
    raster: Path, output_dir: Path = DEFAULT_RESULTS_DIR
) -> CampaignSettings:
    """The documented ``--smoke`` configuration: four methods in about a minute.

    Design note: 50 iterations on a 30 x 30 crop with one seed, run in-process
    (``parallel = 1``).  That is small enough to finish well inside the three
    minutes the smoke test budgets for, while still exercising every method,
    the aggregation, both metric views and all four output documents.
    """
    return CampaignSettings(
        raster=Path(raster),
        tag="smoke",
        methods=DEFAULT_METHODS,
        scenarios=(1,),
        seeds=(0,),
        configs=(parse_scale_config("30:1.0:10"),),
        iterations=50,
        parallel=1,
        output_dir=Path(output_dir),
    )
