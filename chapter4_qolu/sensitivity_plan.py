"""What the sensitivity sweep consists of, and what it will cost.

The published text leaves a set of *documented ambiguities* unsettled, and
each one changes the numbers.  The sweep covers them:

============================  ==========================================
axis                          readings
============================  ==========================================
environment                   whole SE region box / ``E495_N140`` / ``E495_N165``
per-iteration conversion cap  1, 3, 10  (the paper states none)
QOLU reward reading           ``per_parcel`` / ``objective`` / ``team``
P-MADDPG variant              ``published`` / ``reg_only``
LToS variant                  ``ltos`` (learned) / ``fixed_ltos``
MitM variant                  ``mitm`` (fixed) / ``mitm_learned``
============================  ==========================================

Rather than pick one combination and present it as the only reading, the sweep
runs the grid and reports every cell.  Three stages:

**Stage A (screening)** - Scenario 1, seed 0, 1000 iterations.  Scenario 1
weights the three stakeholders equally, so the literal and the weighted
readings of the scenario weights coincide there and the screening does not
depend on which is preferred.  Runs are shared between configurations: a
configuration is a *choice of one run per method*, not a set of runs of its
own, which is what keeps 216 configurations down to 39 runs per environment.

**Stage B (confirmation)** - the top configurations over all four scenarios,
three seeds and *both* consensus readings, scored on all 48 published cells.

**Stage C (deliverable)** - the single best configuration re-run at five
seeds, with the full campaign documents and the figure set.

Design note (``pmaddpg_v1`` is a reference row, not an axis): the unregularised
P-MADDPG actor saturates on some seeds and then approves nothing, so sweeping
it would spend a third of the budget on a configuration whose behaviour is
already known.  It is run once, on the whole region, so the report can show
what it produces; see :attr:`SensitivitySettings.reference_runs`.

Nothing here loads a raster or builds an environment, so the plan and the ETA
can be inspected instantly (``run_sensitivity --dry-run``).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from chapter4_qolu.campaign_plan import estimate_runtime_seconds
from chapter4_qolu.experiment import DEFAULT_RESULTS_DIR
from chapter4_qolu.methods import QOLU_METHOD
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.scan_plan import SINGLE_THREAD_COST_FACTOR
from chapter4_qolu.scenario_worker import RunSpec
from chapter4_qolu.window_scan import ScanWindow, parse_window_label

__all__ = [
    "BASELINE_CAMPAIGN_SUFFIX",
    "CAPS",
    "DEFAULT_ENVIRONMENTS",
    "DEFAULT_WINDOW_LABELS",
    "LTOS_VARIANTS",
    "MITM_VARIANTS",
    "PMADDPG_VARIANTS",
    "REFERENCE_RUNS",
    "REWARD_MODES",
    "ReferenceRun",
    "SensitivityConfiguration",
    "SensitivityPaths",
    "SensitivitySettings",
    "StageBChoice",
    "configurations",
    "estimate_stage_seconds",
    "sensitivity_paths",
]

#: Design note (cap axis): the paper states no per-iteration conversion budget
#: at all; the thesis says 10 and the paper says 11.  1, 3 and 10 span the
#: range from "one parcel a step" to the published figure.
CAPS: tuple[int, ...] = (1, 3, 10)
#: The three readings of ``R_QOLU,i`` (see ``config.QOLURewardConfig``).
REWARD_MODES: tuple[str, ...] = ("per_parcel", "objective", "team")
#: The two P-MADDPG readings that are swept; ``pmaddpg_v1`` is a reference row.
PMADDPG_VARIANTS: tuple[str, ...] = ("pmaddpg", "pmaddpg_reg_only")
LTOS_VARIANTS: tuple[str, ...] = ("ltos", "fixed_ltos")
MITM_VARIANTS: tuple[str, ...] = ("mitm", "mitm_learned")

#: Design note (environments): the paper's Figure 1 plots the *whole* South East
#: England region at ~1 km while the text says "a 50 x 50 grid of parcels, as
#: shown in Figure 1"; the two cannot both be true, so the extent is swept like
#: any other ambiguity.  ``SE_region`` is the whole box (210 x 240 parcels,
#: ~36.5k of them land); the two 50 x 50 windows are the best the Southeast
#: window scan found, carried forward so the square reading is represented by
#: its strongest candidates rather than by an arbitrary crop.
DEFAULT_ENVIRONMENTS: tuple[str, ...] = ("SE_region", "E495_N140", "E495_N165")
#: Alias; the environments *are* the windows.
DEFAULT_WINDOW_LABELS: tuple[str, ...] = DEFAULT_ENVIRONMENTS


@dataclass(frozen=True)
class ReferenceRun:
    """A run made only so the report can show what it did - never swept."""

    environment: str
    cap: int | None
    reward_mode: str
    method: str
    why: str

    def as_record(self) -> dict[str, object]:
        return {
            "environment": self.environment,
            "cap": self.cap,
            "reward_mode": self.reward_mode,
            "method": self.method,
            "why": self.why,
        }


#: Design note (which coordinates a reference run uses): the published cap (10)
#: and the published reward reading (``per_parcel``), one row per environment -
#: the coordinates closest to what the paper actually states, so each row is
#: directly comparable with the swept ``pmaddpg`` cell beside it.
_V1_REASON = (
    "the unregularised P-MADDPG reading (a saturating actor that stops "
    "approving anything on some seeds); kept as one reference row per "
    "environment so the report can show what it produces rather than only "
    "describing it.  It selects the earlier hyper-parameters, not earlier "
    "code: the seeding and noise handling are shared by every variant."
)
REFERENCE_RUNS: tuple[ReferenceRun, ...] = tuple(
    ReferenceRun(
        environment=environment,
        cap=10,
        reward_mode="per_parcel",
        method="pmaddpg_v1",
        why=_V1_REASON,
    )
    for environment in DEFAULT_ENVIRONMENTS
)

#: Suffix of the stage-A campaign holding the methods that take no QOLU reward
#: reading, so they are run once per (window, cap) instead of once per mode.
BASELINE_CAMPAIGN_SUFFIX = "baselines"

#: Methods whose behaviour depends on the QOLU reward reading; see
#: ``scenario_worker._build_baseline`` - only QOLU + BUIA and P-MADDPG are
#: handed a ``QOLURewardConfig`` at all.
_REWARD_SENSITIVE: tuple[str, ...] = (QOLU_METHOD, *PMADDPG_VARIANTS)
_REWARD_FREE: tuple[str, ...] = (*LTOS_VARIANTS, *MITM_VARIANTS)


@dataclass(frozen=True)
class SensitivityConfiguration:
    """One cell of the grid: an environment, a cap, a reward reading, variants."""

    environment: str
    cap: int | None
    reward_mode: str
    pmaddpg: str = "pmaddpg"
    ltos: str = "ltos"
    mitm: str = "mitm"
    conversion_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.reward_mode not in REWARD_MODES:
            raise ValueError(
                f"unknown QOLU reward mode {self.reward_mode!r}; "
                f"expected one of {REWARD_MODES}"
            )
        for name, allowed in (
            (self.pmaddpg, PMADDPG_VARIANTS),
            (self.ltos, LTOS_VARIANTS),
            (self.mitm, MITM_VARIANTS),
        ):
            if name not in allowed:
                raise ValueError(f"unknown variant {name!r}; expected one of {allowed}")
        parse_window_label(self.environment)  # validates the environment label

    @property
    def window(self) -> ScanWindow:
        return parse_window_label(self.environment)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.window.shape

    @property
    def methods(self) -> tuple[str, ...]:
        """The four methods this configuration compares, reference first."""
        return (QOLU_METHOD, self.pmaddpg, self.ltos, self.mitm)

    @property
    def scale(self) -> ScaleConfig:
        rows, cols = self.grid_shape
        return ScaleConfig(
            grid_size=rows,
            conversion_fraction=self.conversion_fraction,
            max_conversions=self.cap,
            grid_width=cols,
        )

    @property
    def label(self) -> str:
        """Filesystem-safe name, e.g. ``SE_region_cap1_per_parcel_pmaddpg_ltos_mitm``."""
        cap = "capnone" if self.cap is None else f"cap{self.cap}"
        return (
            f"{self.environment}_{cap}_{self.reward_mode}"
            f"_{self.pmaddpg}_{self.ltos}_{self.mitm}"
        )

    @property
    def description(self) -> str:
        cap = "no cap" if self.cap is None else f"cap {self.cap}"
        rows, cols = self.grid_shape
        return (
            f"`{self.environment}` ({rows}x{cols}), {cap}, reward reading "
            f"`{self.reward_mode}`, `{self.pmaddpg}` / `{self.ltos}` / "
            f"`{self.mitm}`"
        )

    def as_record(self) -> dict[str, object]:
        return {
            "label": self.label,
            "description": self.description,
            "environment": self.environment,
            "grid_shape": list(self.grid_shape),
            "cap": self.cap,
            "reward_mode": self.reward_mode,
            "pmaddpg": self.pmaddpg,
            "ltos": self.ltos,
            "mitm": self.mitm,
            "configuration": self.scale.label,
        }


def configurations(
    environments: Sequence[str] = DEFAULT_ENVIRONMENTS,
    caps: Sequence[int | None] = CAPS,
    reward_modes: Sequence[str] = REWARD_MODES,
    pmaddpg_variants: Sequence[str] = PMADDPG_VARIANTS,
    ltos_variants: Sequence[str] = LTOS_VARIANTS,
    mitm_variants: Sequence[str] = MITM_VARIANTS,
    conversion_fraction: float = 1.0,
) -> tuple[SensitivityConfiguration, ...]:
    """Every cell of the grid, in a deterministic order."""
    return tuple(
        SensitivityConfiguration(
            environment=environment,
            cap=cap,
            reward_mode=mode,
            pmaddpg=pmaddpg,
            ltos=ltos,
            mitm=mitm,
            conversion_fraction=conversion_fraction,
        )
        for environment, cap, mode, pmaddpg, ltos, mitm in itertools.product(
            environments,
            caps,
            reward_modes,
            pmaddpg_variants,
            ltos_variants,
            mitm_variants,
        )
    )


@dataclass(frozen=True)
class SensitivitySettings:
    """What the sweep searches and how hard."""

    raster: Path
    tag: str = "20260924"
    environments: tuple[str, ...] = DEFAULT_ENVIRONMENTS
    caps: tuple[int | None, ...] = CAPS
    reward_modes: tuple[str, ...] = REWARD_MODES
    pmaddpg_variants: tuple[str, ...] = PMADDPG_VARIANTS
    ltos_variants: tuple[str, ...] = LTOS_VARIANTS
    mitm_variants: tuple[str, ...] = MITM_VARIANTS
    reference_runs: tuple[ReferenceRun, ...] = REFERENCE_RUNS
    conversion_fraction: float = 1.0
    iterations: int = 1000
    stakeholders: int = 3
    stage_a_seed: int = 0
    stage_b_seeds: tuple[int, ...] = (0, 1, 2)
    stage_b_scenarios: tuple[int, ...] = (1, 2, 3, 4)
    #: Design note: three configurations go through to stage B, for the same
    #: reason the window scan confirms three windows - enough to show whether
    #: the screening winner is a fluke, few enough to cost about what the
    #: screening did.
    stage_b_configurations: int = 3
    #: Design note: five seeds in stage C.  The published standard errors are
    #: 0.01-0.03 and the paper never says how many runs they come from; five is
    #: the smallest count that makes a paired t-test worth quoting, and the
    #: report says what SE the campaign actually reaches at five.
    stage_c_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    max_distance_km: float = 10.0
    decimals: int = 3
    parallel: int = 16
    resume: bool = False
    region_mask: Path | None = None
    output_dir: Path = field(default=DEFAULT_RESULTS_DIR)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raster", Path(self.raster))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in (
            "environments",
            "caps",
            "reward_modes",
            "pmaddpg_variants",
            "ltos_variants",
            "mitm_variants",
            "reference_runs",
            "stage_b_seeds",
            "stage_b_scenarios",
            "stage_c_seeds",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.region_mask is not None:
            object.__setattr__(self, "region_mask", Path(self.region_mask))
        if not self.environments:
            raise ValueError("at least one environment is required")
        if not self.caps:
            raise ValueError("at least one conversion cap is required")
        if self.stage_b_configurations < 1:
            raise ValueError("stage_b_configurations must be positive")
        if not self.stage_b_seeds:
            raise ValueError("at least one stage-B seed is required")

    @property
    def window_labels(self) -> tuple[str, ...]:
        """Alias: the environments *are* the windows."""
        return self.environments

    @property
    def windows(self) -> tuple[ScanWindow, ...]:
        """The pinned environments, resolved from their labels."""
        return tuple(parse_window_label(label) for label in self.environments)

    def configurations(self) -> tuple[SensitivityConfiguration, ...]:
        return configurations(
            environments=self.environments,
            caps=self.caps,
            reward_modes=self.reward_modes,
            pmaddpg_variants=self.pmaddpg_variants,
            ltos_variants=self.ltos_variants,
            mitm_variants=self.mitm_variants,
            conversion_fraction=self.conversion_fraction,
        )

    def scales_for(self, window: ScanWindow) -> tuple[ScaleConfig, ...]:
        rows, cols = window.shape
        return tuple(
            ScaleConfig(
                grid_size=rows,
                conversion_fraction=self.conversion_fraction,
                max_conversions=cap,
                grid_width=cols,
            )
            for cap in self.caps
        )


@dataclass(frozen=True)
class SensitivityPaths:
    """Where a sweep writes."""

    root: Path
    stage_a: Path
    stage_b: Path
    stage_b2: Path
    stage_c: Path
    summary: Path
    report: Path
    figures: Path


def sensitivity_paths(settings: SensitivitySettings) -> SensitivityPaths:
    root = Path(settings.output_dir) / f"sensitivity_{settings.tag}"
    return SensitivityPaths(
        root=root,
        stage_a=root / "stageA",
        stage_b=root / "stageB",
        stage_b2=root / "stageB2",
        stage_c=root / "stageC",
        summary=root / "sensitivity_summary.json",
        report=root / "sensitivity.md",
        figures=root / "figures",
    )


@dataclass(frozen=True)
class StageBChoice:
    """One configuration going through to the confirmation stage."""

    rank: int
    configuration: SensitivityConfiguration
    window: ScanWindow

    @property
    def base_tag(self) -> str:
        return f"b{self.rank}_{self.configuration.label}"

    def as_record(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "configuration": self.configuration.as_record(),
            "window": self.window.as_record(),
            "base_tag": self.base_tag,
        }


def estimate_stage_seconds(specs: Sequence[RunSpec]) -> float:
    """Estimated *serial* CPU cost of a set of runs, single-thread corrected.

    The same correction :mod:`chapter4_qolu.scan_plan` applies, and for the
    same reason: ``campaign_plan.METHOD_COST_SECONDS`` was measured with torch
    free to use every core, while a sweep worker is pinned to one thread.
    """
    return estimate_runtime_seconds(specs) * SINGLE_THREAD_COST_FACTOR
