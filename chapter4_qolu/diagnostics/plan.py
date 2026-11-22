"""What one diagnostics run is, where it writes, and where its agents come from.

Split out of :mod:`chapter4_qolu.run_distribution_diagnostics` for the same
reason :mod:`chapter4_qolu.sensitivity_plan` is split out of the sweep driver:
"what is being asked for" is a different thing from "doing it", and only the
former needs to be readable from a test or from another driver.

The settings deliberately hang off :class:`~chapter4_qolu.scenario_worker.RunSpec`
rather than restating it.  A diagnostic is a diagnostic *of a run*, and the run
it is of has to be described in exactly the same terms a campaign describes it
in - the same window, the same scale configuration, the same reward reading -
or the numbers are about a different experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from chapter4_qolu.agent.buia_agent import BUIAAgent
from chapter4_qolu.agent.checkpoint import agent_checkpoint_paths
from chapter4_qolu.campaign_plan import estimate_runtime_seconds, format_duration
from chapter4_qolu.config import SCENARIOS, BUIAConfig
from chapter4_qolu.data.lcm_loader import (
    load_lcm_raster,
    resolve_window_origin,
    select_region,
)
from chapter4_qolu.diagnostics.records import HEADLINE_TARGET
from chapter4_qolu.diagnostics.returns import TARGET_DESCRIPTIONS
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.experiment import DEFAULT_RESULTS_DIR
from chapter4_qolu.methods import QOLU_METHOD, QOLU_METHOD_VARIANTS
from chapter4_qolu.reporting.paths import ScaleConfig, campaign_directory
from chapter4_qolu.scenario_worker import RunSpec, buia_proposal_limit
from chapter4_qolu.window_scan import ScanWindow, parse_window_label

__all__ = [
    "AGENT_DIR_SUFFIX",
    "DEFAULT_DECIMALS",
    "DIAGNOSTICS_PREFIX",
    "DOCUMENT_STEM",
    "DiagnosticsPaths",
    "DiagnosticsSettings",
    "agent_source",
    "build_environment",
    "campaign_agent_directory",
    "diagnostics_paths",
    "investor_for",
    "is_complete",
    "run_plan",
    "scale_for_window",
]

#: Folder a campaign run's persisted stakeholders live in, beside its record.
AGENT_DIR_SUFFIX: str = ".agents"
#: Prefix of the results folder, so diagnostics runs sort together.
DIAGNOSTICS_PREFIX: str = "distribution_diagnostics_"
#: Stem of both documents, so the pair is obviously one thing.
DOCUMENT_STEM: str = "distribution_diagnostics"
#: Design note: four decimals, not the tables' three.  A CRPS skill score and a
#: boundary mass are both small numbers whose third decimal carries the claim.
DEFAULT_DECIMALS: int = 4


@dataclass(frozen=True)
class DiagnosticsPaths:
    """Where one diagnostics run writes."""

    root: Path
    summary: Path
    report: Path
    figures: Path
    agents: Path


@dataclass(frozen=True)
class DiagnosticsSettings:
    """Everything one diagnostics run needs."""

    raster: Path
    tag: str = "main"
    method: str = QOLU_METHOD
    scenario: int = 1
    seed: int = 0
    scale: ScaleConfig = field(
        default_factory=lambda: ScaleConfig(
            grid_size=50, conversion_fraction=1.0, max_conversions=10
        )
    )
    window_label: str | None = None
    stakeholders: int = 3
    #: Training iterations, used only when no agents are already on disk.
    iterations: int = 1000
    #: Length of the evaluation rollout the statistics are computed over.
    eval_iterations: int = 200
    qolu_reward_mode: str = "per_parcel"
    scale_by_scenario_weight: bool = False
    weighted_consensus: bool = False
    #: ``t_n`` of Algorithm 1's "every ``t_n`` steps reset ``theta^- = theta``",
    #: which Table 4.1 never gives.  ``None`` keeps
    #: :class:`~chapter4_qolu.config.QOLUConfig`'s default; a value moves this
    #: run's learner off it through the single override route in
    #: :func:`~chapter4_qolu.scenario_worker.qolu_config_for`.  It is settable
    #: per run precisely because the default is a choice rather than a
    #: published number, and a value longer than a run's gradient budget never
    #: fires at all - see :mod:`chapter4_qolu.hparam_inert`.
    target_sync_interval: int | None = None
    #: The C51 atom support, Table 4.1's "51 atoms over [-200, 200]".  Unlike
    #: ``t_n`` these three **are** published, so ``None`` - the package default
    #: - is what reproduces the chapter and nothing here moves it by accident.
    #: They are settable per run because the published support is ~270 times
    #: wider than this environment's achievable return (the ``per_parcel``
    #: reward lies in [-0.5, 0.5], so at gamma = 0.99 and n_step = 3 the whole
    #: n-step reward sum spans 2.97 against an atom spacing of delta_z = 8),
    #: and whether the categorical head can resolve a per-parcel difference at
    #: all is a question only an experiment can answer.  A run that moves them
    #: is measuring its *own* support, not the chapter's, and says so in its
    #: record.
    n_atoms: int | None = None
    v_min: float | None = None
    v_max: float | None = None
    #: How many ``learn()`` calls one training iteration fires.  ``None`` keeps
    #: :class:`~chapter4_qolu.config.QOLUConfig`'s default of 1, one
    #: ``learn()`` per iteration; the gradient budget is nowhere in
    #: the published text, and at 1000 iterations it samples each stored
    #: transition about five times in a whole run.
    learn_steps_per_iteration: int | None = None
    #: Read persisted stakeholders from a campaign directory instead of training.
    campaign: Path | None = None
    #: Read persisted stakeholders from one directory instead of training.
    checkpoint: Path | None = None
    #: Attach a bottom-up investor to the evaluation rollout.
    investor: bool = True
    headline_target: str = HEADLINE_TARGET
    #: Seed of the PIT randomisation, so a reported KS statistic is reproducible.
    diagnostics_seed: int = 0
    decimals: int = DEFAULT_DECIMALS
    resume: bool = False
    torch_threads: int = 1
    output_dir: Path = DEFAULT_RESULTS_DIR

    def __post_init__(self) -> None:
        if self.method not in QOLU_METHOD_VARIANTS:
            raise ValueError(
                f"{self.method!r} is not a QOLU method; expected one of "
                f"{tuple(QOLU_METHOD_VARIANTS)}"
            )
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {self.scenario}; expected 1-4")
        if self.eval_iterations < 1:
            raise ValueError("eval_iterations must be positive")
        if self.target_sync_interval is not None and self.target_sync_interval < 1:
            raise ValueError("target_sync_interval must be positive")
        if self.n_atoms is not None and self.n_atoms < 2:
            raise ValueError("n_atoms must be at least two")
        if (
            self.v_min is not None
            and self.v_max is not None
            and self.v_max <= self.v_min
        ):
            raise ValueError("v_max must be greater than v_min")
        if (
            self.learn_steps_per_iteration is not None
            and self.learn_steps_per_iteration < 1
        ):
            raise ValueError("learn_steps_per_iteration must be positive")
        if self.headline_target not in TARGET_DESCRIPTIONS:
            raise ValueError(
                f"unknown headline target {self.headline_target!r}; expected "
                f"one of {sorted(TARGET_DESCRIPTIONS)}"
            )
        window = self.window()
        if window is not None and window.shape != self.scale.shape:
            raise ValueError(
                f"window {window.label} is {window.shape} but the configuration "
                f"is {self.scale.shape}; pin one or make them agree"
            )

    def window(self) -> ScanWindow | None:
        """The pinned window, or ``None`` for the automatic densest-land crop."""
        if self.window_label is None:
            return None
        return parse_window_label(self.window_label)

    @property
    def qolu_overrides(self) -> dict[str, Any]:
        """The :class:`QOLUConfig` fields this run moves, for ``qolu_config_for``.

        Empty when nothing was asked for, so a run that names no override takes
        the package defaults through the default code path.  A field
        left at ``None`` is absent from the mapping rather than present with a
        restated default, so the record and the config can never disagree about
        which settings this run actually chose.
        """
        asked: dict[str, Any] = {
            "target_sync_interval": (
                None
                if self.target_sync_interval is None
                else int(self.target_sync_interval)
            ),
            "n_atoms": None if self.n_atoms is None else int(self.n_atoms),
            "v_min": None if self.v_min is None else float(self.v_min),
            "v_max": None if self.v_max is None else float(self.v_max),
            "learn_steps_per_iteration": (
                None
                if self.learn_steps_per_iteration is None
                else int(self.learn_steps_per_iteration)
            ),
        }
        return {name: value for name, value in asked.items() if value is not None}

    def run_spec(self, output_path: Path) -> RunSpec:
        """The campaign run this diagnostic is *of*, in the campaign's own terms."""
        window = self.window()
        return RunSpec(
            raster=self.raster,
            method=self.method,
            scenario=self.scenario,
            seed=self.seed,
            scale=self.scale,
            output_path=output_path,
            iterations=self.iterations,
            stakeholders=self.stakeholders,
            qolu_reward_mode=self.qolu_reward_mode,
            scale_by_scenario_weight=self.scale_by_scenario_weight,
            weighted_consensus=self.weighted_consensus,
            window_bng_km=None if window is None else window.bng_top_left_km,
            window_label=None if window is None else window.label,
            torch_threads=self.torch_threads,
        )

    def as_record(self) -> dict[str, Any]:
        """What produced these numbers, for the summary JSON."""
        return {
            "method": self.method,
            "scenario": self.scenario,
            "weights": {
                "land_use": SCENARIOS[self.scenario].land_use,
                "woodland": SCENARIOS[self.scenario].woodland,
                "urban": SCENARIOS[self.scenario].urban,
            },
            "seed": self.seed,
            "stakeholders": self.stakeholders,
            "window": self.window_label,
            "configuration": self.scale.as_record(),
            "raster": Path(self.raster).name,
            "qolu_reward_mode": self.qolu_reward_mode,
            "scale_by_scenario_weight": bool(self.scale_by_scenario_weight),
            "weighted_consensus": bool(self.weighted_consensus),
            "training_iterations": self.iterations,
            "evaluation_iterations": self.eval_iterations,
            "diagnostics_seed": self.diagnostics_seed,
            # ``None`` means "whatever config.py says", which is the honest
            # record: it names the run's provenance rather than a number that
            # would silently stop matching if the default ever moved.  The same
            # convention covers the atom support and the gradient budget, so
            # `null` here reads "the published 51 atoms over [-200, 200], one
            # learn() per iteration" and a number reads "this run chose it".
            "target_sync_interval": self.target_sync_interval,
            "n_atoms": self.n_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "learn_steps_per_iteration": self.learn_steps_per_iteration,
        }


def diagnostics_paths(settings: DiagnosticsSettings) -> DiagnosticsPaths:
    """Result-file layout of one diagnostics run."""
    root = Path(settings.output_dir) / f"{DIAGNOSTICS_PREFIX}{settings.tag}"
    return DiagnosticsPaths(
        root=root,
        summary=root / f"{DOCUMENT_STEM}.json",
        report=root / f"{DOCUMENT_STEM}.md",
        figures=root / "figures",
        agents=root / "agents",
    )


def scale_for_window(label: str | None, default: ScaleConfig) -> ScaleConfig:
    """Take the extent from a window label when no configuration was given."""
    if label is None:
        return default
    rows, columns = parse_window_label(label).shape
    return replace(default, grid_size=rows, grid_width=columns)


def campaign_agent_directory(settings: DiagnosticsSettings) -> Path | None:
    """Where a campaign run would have persisted its stakeholders, if it did."""
    if settings.campaign is None:
        return None
    campaign = Path(settings.campaign)
    if not campaign.is_dir():
        campaign = campaign_directory(settings.output_dir, str(settings.campaign))
    return (
        campaign
        / settings.scale.label
        / f"scenario{settings.scenario}"
        / f"{settings.method}_seed{settings.seed}{AGENT_DIR_SUFFIX}"
    )


def agent_source(settings: DiagnosticsSettings, paths: DiagnosticsPaths) -> Path | None:
    """The checkpoint directory to read, or ``None`` when a run must be trained.

    Search order: an explicit ``--checkpoint``, then the campaign cell, then
    this diagnostics run's own ``agents/`` folder but **only** under
    ``--resume``.  Without ``--resume`` a previous run's agents are ignored, so
    re-running the driver re-trains rather than silently diagnosing something
    older than the code.
    """
    for candidate in (
        settings.checkpoint,
        campaign_agent_directory(settings),
        paths.agents if settings.resume else None,
    ):
        if candidate is not None and agent_checkpoint_paths(candidate):
            return Path(candidate)
    return None


def is_complete(path: Path) -> bool:
    """``True`` when ``path`` holds a finished diagnostics summary."""
    if not Path(path).is_file():
        return False
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    return isinstance(record, dict) and "verdict" in record


def build_environment(spec: RunSpec) -> UrbanLandUseEnv:
    """The environment one run was carried out in, rebuilt from its spec."""
    raster = load_lcm_raster(spec.raster)
    origin = resolve_window_origin(
        raster,
        spec.grid_shape,
        origin=spec.window_origin,
        bng_origin_km=spec.window_bng_km,
    )
    grid = select_region(raster, shape=spec.grid_shape, origin=origin)
    return UrbanLandUseEnv(
        grid, config=spec.env_config(), weights=SCENARIOS[spec.scenario]
    )


def investor_for(env: UrbanLandUseEnv, spec: RunSpec) -> BUIAAgent:
    """A bottom-up investor matching ``env``, seeded like the run."""
    return BUIAAgent(
        BUIAConfig(
            observation_channels=env.buia_observation_channels,
            observation_window=env.config.buia_observation_window,
            max_proposals=buia_proposal_limit(env.config),
        ),
        seed=spec.seed,
    )


def run_plan(
    settings: DiagnosticsSettings, spec: RunSpec, paths: DiagnosticsPaths
) -> dict[str, Any]:
    """What a run would do, with its ETA - the body of ``--dry-run``."""
    source = agent_source(settings, paths)
    seconds = 0.0 if source is not None else estimate_runtime_seconds([spec])
    return {
        "tag": settings.tag,
        "run": settings.as_record(),
        "training_iterations": settings.iterations,
        "evaluation_iterations": settings.eval_iterations,
        "agents": "checkpoint" if source is not None else "train",
        "agent_directory": str(source if source is not None else paths.agents),
        "estimated_training_seconds": round(seconds, 1),
        "estimated_training": format_duration(seconds),
        "outputs": [str(paths.summary), str(paths.report), str(paths.figures)],
    }
