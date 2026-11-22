"""What the hyper-parameter sensitivity analysis consists of, and what it costs.

One *configuration under test* - an extent, a conversion cap, a reward reading,
a set of scenarios and a set of seeds - is run once as the **baseline**, and
then once more per perturbation, with a single hyper-parameter moved +/-20 %
(:mod:`chapter4_qolu.hparam_sensitivity`).  Every stage is expressed as an
ordinary :class:`~chapter4_qolu.campaign_plan.CampaignSettings`, exactly as the
window scan and the configuration sweep do, so the result layout, ``--resume``
and the campaign documents are the ordinary campaign machinery.

Design note (which methods a perturbation re-runs).  All eleven perturbed
quantities belong to the QOLU learner or to the shared environment, never to a
baseline's own optimiser:

* an **environment**-side perturbation (the sampled-parcel count, the
  observation window) changes what *every* method sees, so all four are re-run;
* an **agent**-side one changes only ``QOLUConfig``, so P-MADDPG, LToS and MitM
  would reproduce their baseline records bit for bit - same seed, same crop,
  same environment settings.  They are therefore not re-run at all and the
  report reads them from the baseline campaign, the same economy
  :func:`chapter4_qolu.sensitivity_stages.stage_a_campaigns` makes for the
  reward-free baselines.

That also fixes the reading of the result: an agent-side row asks *does the
published ordering survive detuning QOLU itself*, holding the comparison
methods at their published settings, which is the conservative direction for a
claim that QOLU wins.

Nothing here loads a raster or builds an environment, so the plan and the ETA
can be inspected instantly (``run_hparam_sensitivity --dry-run``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chapter4_qolu.campaign_plan import (
    CampaignSettings,
    campaign_runs,
    estimate_runtime_seconds,
)
from chapter4_qolu.config import SCENARIOS
from chapter4_qolu.experiment import DEFAULT_RESULTS_DIR
from chapter4_qolu.hparam_sensitivity import (
    PERTURBATION_FRACTION,
    Perturbation,
    hyperparameter_names,
    perturbations,
    resolve_hyperparameters,
)
from chapter4_qolu.hparam_worker import HParamRunSpec, as_hparam_spec
from chapter4_qolu.methods import METHOD_KEYS, PUBLISHED_METHODS, QOLU_METHOD
from chapter4_qolu.reporting.documents import DEFAULT_DECIMALS
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.reporting.stats import MIN_PAIRED_SAMPLES
from chapter4_qolu.scan_plan import SINGLE_THREAD_COST_FACTOR
from chapter4_qolu.window_scan import parse_window_label

__all__ = [
    "BASELINE_TAG",
    "DEFAULT_SCALE",
    "HParamPaths",
    "HParamSettings",
    "all_runs",
    "baseline_campaign",
    "baseline_runs",
    "effective_perturbations",
    "estimate_seconds",
    "hparam_paths",
    "perturbation_campaign",
    "perturbation_runs",
    "settings_perturbations",
]

#: Campaign tag of the unperturbed configuration every row is compared with.
BASELINE_TAG = "baseline"

#: Design note (the configuration under test): the published reading - a
#: 50 x 50 km extent, whole-parcel conversion, the thesis's cap of ten
#: conversions an iteration.  It is the configuration the chapter's tables
#: describe, so it is the one whose comparative conclusions are being tested;
#: ``--config`` moves it.
DEFAULT_SCALE = ScaleConfig(grid_size=50, conversion_fraction=1.0, max_conversions=10)


@dataclass(frozen=True)
class HParamSettings:
    """The configuration under test and how hard it is perturbed."""

    raster: Path
    tag: str = "hparam"
    #: Names from :data:`chapter4_qolu.hparam_sensitivity.KEY_HYPERPARAMETERS`;
    #: ``None`` means all of them.
    hyperparameters: tuple[str, ...] | None = None
    fraction: float = PERTURBATION_FRACTION
    methods: tuple[str, ...] = PUBLISHED_METHODS
    #: Design note: Scenario 1 only, by default.  It weights the three
    #: stakeholders equally, so the literal and the weighted readings of the
    #: scenario weights coincide there and the perturbation is not confounded
    #: with a weighting choice; ``--scenarios`` runs the other three.
    scenarios: tuple[int, ...] = (1,)
    #: Design note: three seeds, the fewest that make a ``mean +/- SE`` and a
    #: paired t-test worth quoting (:data:`reporting.stats.MIN_PAIRED_SAMPLES`),
    #: and the count that keeps 21 effective perturbations affordable.
    seeds: tuple[int, ...] = (0, 1, 2)
    iterations: int = 1000
    scale: ScaleConfig = DEFAULT_SCALE
    stakeholders: int = 3
    qolu_reward_mode: str = "per_parcel"
    scale_by_scenario_weight: bool = False
    weighted_consensus: bool = False
    max_distance_km: float = 10.0
    #: Optional pinned window label, e.g. ``E495_N140``; ``None`` keeps the
    #: loader's automatic densest-land crop.
    window: str | None = None
    region_mask: Path | None = None
    decimals: int = DEFAULT_DECIMALS
    parallel: int = 16
    resume: bool = False
    output_dir: Path = field(default=DEFAULT_RESULTS_DIR)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raster", Path(self.raster))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "methods", tuple(self.methods))
        object.__setattr__(self, "scenarios", tuple(int(k) for k in self.scenarios))
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))
        if self.hyperparameters is not None:
            object.__setattr__(self, "hyperparameters", tuple(self.hyperparameters))
        if self.region_mask is not None:
            object.__setattr__(self, "region_mask", Path(self.region_mask))
        if not self.methods:
            raise ValueError("at least one method is required")
        unknown = [name for name in self.methods if name not in METHOD_KEYS]
        if unknown:
            raise ValueError(f"unknown method(s) {unknown}; expected {METHOD_KEYS}")
        if QOLU_METHOD not in self.methods:
            raise ValueError(
                f"{QOLU_METHOD!r} has to be among the methods: the perturbed "
                "hyper-parameters are its own"
            )
        if not self.scenarios:
            raise ValueError("at least one scenario is required")
        unknown_scenarios = [k for k in self.scenarios if k not in SCENARIOS]
        if unknown_scenarios:
            raise ValueError(f"unknown scenario(s) {unknown_scenarios}; expected 1-4")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        if len(self.seeds) < MIN_PAIRED_SAMPLES:
            raise ValueError(
                f"at least three seeds are required, got {len(self.seeds)}: a "
                "mean +/- SE and a paired t-test are what the rows report"
            )
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if not 0.0 < self.fraction < 1.0:
            raise ValueError("fraction must lie in (0, 1)")
        if self.parallel < 1:
            raise ValueError("parallel must be at least one")
        resolve_hyperparameters(self.hyperparameter_names)

    @property
    def hyperparameter_names(self) -> tuple[str, ...]:
        """The names under test; every key one unless ``--hyperparameters``."""
        if self.hyperparameters is not None:
            return tuple(self.hyperparameters)
        return hyperparameter_names()

    @property
    def window_bng_km(self) -> tuple[float, float] | None:
        """North-west corner of the pinned window, or ``None``."""
        if not self.window:
            return None
        window = parse_window_label(self.window, size_km=self.scale.width)
        return tuple(float(value) for value in window.bng_top_left_km)  # type: ignore[return-value]

    def as_record(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "raster": Path(self.raster).name,
            "fraction": self.fraction,
            "hyperparameters": list(self.hyperparameter_names),
            "methods": list(self.methods),
            "scenarios": list(self.scenarios),
            "seeds": list(self.seeds),
            "iterations": self.iterations,
            "configuration": self.scale.as_record(),
            "stakeholders": self.stakeholders,
            "qolu_reward_mode": self.qolu_reward_mode,
            "scale_by_scenario_weight": bool(self.scale_by_scenario_weight),
            "weighted_consensus": bool(self.weighted_consensus),
            "max_distance_km": self.max_distance_km,
            "window": self.window,
            "region_mask": (
                Path(self.region_mask).name if self.region_mask is not None else None
            ),
            "parallel": self.parallel,
        }


@dataclass(frozen=True)
class HParamPaths:
    """Where one analysis writes."""

    root: Path
    runs: Path
    summary: Path
    report: Path
    runs_csv: Path


def hparam_paths(settings: HParamSettings) -> HParamPaths:
    root = Path(settings.output_dir) / f"hparam_{settings.tag}"
    return HParamPaths(
        root=root,
        runs=root / "runs",
        summary=root / "hparam_summary.json",
        report=root / "hparam_sensitivity.md",
        runs_csv=root / "hparam_runs.csv",
    )


def settings_perturbations(settings: HParamSettings) -> tuple[Perturbation, ...]:
    """Every perturbation this analysis lists, no-ops included."""
    return perturbations(
        resolve_hyperparameters(settings.hyperparameter_names), settings.fraction
    )


def effective_perturbations(settings: HParamSettings) -> tuple[Perturbation, ...]:
    """The perturbations that actually move a value, i.e. the ones that run."""
    return tuple(item for item in settings_perturbations(settings) if not item.is_no_op)


def _campaign(
    settings: HParamSettings, tag: str, methods: Sequence[str]
) -> CampaignSettings:
    paths = hparam_paths(settings)
    return CampaignSettings(
        raster=settings.raster,
        tag=tag,
        methods=tuple(methods),
        scenarios=settings.scenarios,
        seeds=settings.seeds,
        configs=(settings.scale,),
        iterations=settings.iterations,
        stakeholders=settings.stakeholders,
        qolu_reward_mode=settings.qolu_reward_mode,
        scale_by_scenario_weight=settings.scale_by_scenario_weight,
        weighted_consensus=settings.weighted_consensus,
        max_distance_km=settings.max_distance_km,
        parallel=settings.parallel,
        resume=settings.resume,
        window_bng_km=settings.window_bng_km,
        window_label=settings.window,
        region_mask=settings.region_mask,
        decimals=settings.decimals,
        output_dir=paths.runs,
    )


def baseline_campaign(settings: HParamSettings) -> CampaignSettings:
    """The unperturbed configuration under test: every method, every seed."""
    return _campaign(settings, BASELINE_TAG, settings.methods)


def perturbation_methods(
    settings: HParamSettings, perturbation: Perturbation
) -> tuple[str, ...]:
    """Which methods one perturbation has to re-run (see the module docstring)."""
    if perturbation.target == "environment":
        return tuple(settings.methods)
    return (QOLU_METHOD,)


def perturbation_campaign(
    settings: HParamSettings, perturbation: Perturbation
) -> CampaignSettings:
    """The campaign one perturbation runs, tagged with its label."""
    return _campaign(
        settings, perturbation.label, perturbation_methods(settings, perturbation)
    )


def baseline_runs(settings: HParamSettings) -> tuple[HParamRunSpec, ...]:
    """Every run of the baseline configuration."""
    return tuple(
        as_hparam_spec(spec, None)
        for spec in campaign_runs(baseline_campaign(settings))
    )


def perturbation_runs(
    settings: HParamSettings, perturbation: Perturbation
) -> tuple[HParamRunSpec, ...]:
    """Every run of one perturbation; empty when it is arithmetically a no-op."""
    if perturbation.is_no_op:
        return ()
    campaign = perturbation_campaign(settings, perturbation)
    return tuple(as_hparam_spec(spec, perturbation) for spec in campaign_runs(campaign))


def all_runs(settings: HParamSettings) -> tuple[HParamRunSpec, ...]:
    """The whole analysis: the baseline, then every perturbation in order."""
    specs: dict[Path, HParamRunSpec] = {}
    for spec in baseline_runs(settings):
        specs.setdefault(spec.output_path, spec)
    for perturbation in settings_perturbations(settings):
        for spec in perturbation_runs(settings, perturbation):
            specs.setdefault(spec.output_path, spec)
    return tuple(specs.values())


def estimate_seconds(specs: Sequence[HParamRunSpec]) -> float:
    """Estimated *serial* CPU cost of a set of runs, single-thread corrected.

    The same correction :mod:`chapter4_qolu.scan_plan` and
    :mod:`chapter4_qolu.sensitivity_plan` apply, and for the same reason:
    ``campaign_plan.METHOD_COST_SECONDS`` was measured with torch free to use
    every core, while a worker here is pinned to one thread.
    """
    return estimate_runtime_seconds(specs) * SINGLE_THREAD_COST_FACTOR
