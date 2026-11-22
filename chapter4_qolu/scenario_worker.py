"""Execution of a single campaign run - one method, one cell, one seed.

A *run* is one (configuration, scenario, method, seed) cell of the campaign
driven by :mod:`chapter4_qolu.run_scenarios`.  This module is deliberately
separate from that driver because it is what a worker process imports: it holds
the picklable :class:`RunSpec` and the single entry point
:func:`execute_run`, and it never decides *which* runs to perform.

Comparability is the whole point, so every method in a cell gets the same
deterministic raster crop, the same :class:`~chapter4_qolu.config.EnvConfig`
and the same seed.  ``initial_grid_digest`` is written into every record so that
can be checked after the fact rather than assumed.

The crop is either pinned - by raster ``(row, col)`` or by the British National
Grid coordinates of its top-left corner, which is what the Southeast window scan
of :mod:`chapter4_qolu.run_window_scan` uses - or chosen automatically by
``select_region_of_interest`` (the densest land window for a given
``grid_size``).  Either way the window that was *actually* used is written into
the record.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chapter4_qolu.agent.buia_agent import BUIAAgent, buia_proposal_limit
from chapter4_qolu.agent.qolu_agent import QOLUAgent
from chapter4_qolu.agent.qolu_variants import qolu_variant_settings
from chapter4_qolu.baselines.base import BaselineMethod, BaselineReport, run_baseline
from chapter4_qolu.baselines.ltos import LToSMethod, fixed_ltos
from chapter4_qolu.baselines.mitm import MitMMethod, learned_mitm
from chapter4_qolu.baselines.no_regulator import NoRegulatorMethod
from chapter4_qolu.baselines.pmaddpg import PMADDPGMethod
from chapter4_qolu.config import (
    SCENARIOS,
    BUIAConfig,
    EnvConfig,
    QOLUConfig,
    QOLURewardConfig,
)
from chapter4_qolu.data.lcm_loader import (
    load_lcm_raster,
    resolve_window_origin,
    select_region,
)
from chapter4_qolu.data.region_mask import apply_region_mask, load_region_mask
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.methods import (
    METHOD_KEYS,
    NO_REGULATOR_METHOD,
    PMADDPG_METHOD_VARIANTS,
    QOLU_METHOD,
    QOLU_METHOD_VARIANTS,
)
from chapter4_qolu.reporting.paths import ScaleConfig
from chapter4_qolu.run_record import config_record, learner_record, run_record
from chapter4_qolu.runner import JointTrainingReport, train_joint

__all__ = [
    "METHOD_KEYS",
    "QOLU_METHOD",
    "RunSpec",
    "buia_proposal_limit",
    "execute_run",
    "grid_digest",
    "qolu_config_for",
]


@dataclass(frozen=True)
class RunSpec:
    """Everything one campaign run needs, in a picklable form."""

    raster: Path
    method: str
    scenario: int
    seed: int
    scale: ScaleConfig
    output_path: Path
    iterations: int = 1000
    stakeholders: int = 3
    qolu_reward_mode: str = "per_parcel"
    #: Optional routes by which the scenario weights reach the top-down agents
    #: (see :class:`~chapter4_qolu.config.QOLURewardConfig`).  Both default to
    #: ``False``, the published reading; a campaign that turns them on belongs
    #: under its own ``--tag``.
    scale_by_scenario_weight: bool = False
    weighted_consensus: bool = False
    max_distance_km: float = 10.0
    #: Optional pinned raster window: ``(row, col)`` of its top-left parcel.
    window_origin: tuple[int, int] | None = None
    #: Optional pinned raster window: projected-CRS ``(easting, northing)`` in
    #: kilometres of its top-left (north-west) corner.  Mutually exclusive with
    #: ``window_origin``; both ``None`` keeps the automatic densest-land window.
    window_bng_km: tuple[float, float] | None = None
    #: Human-readable name of the pinned window, e.g. ``E495_N140``.
    window_label: str | None = None
    #: Optional polygon boundary clipping the window (see
    #: :mod:`chapter4_qolu.data.region_mask`).  ``None`` keeps the whole box.
    region_mask: Path | None = None
    #: Design note: one torch thread per worker.  The campaign parallelises
    #: across *runs*, so letting each run grab every core would oversubscribe
    #: the machine and make the wall time unpredictable.
    torch_threads: int = 1

    def __post_init__(self) -> None:
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
        if self.region_mask is not None:
            object.__setattr__(self, "region_mask", Path(self.region_mask))
        if self.window_origin is not None and self.window_bng_km is not None:
            raise ValueError(
                "pin a window by (row, col) or by BNG coordinates, not both: "
                f"{self.window_origin!r} and {self.window_bng_km!r}"
            )
        if self.method not in METHOD_KEYS:
            raise ValueError(
                f"unknown method {self.method!r}; expected one of {METHOD_KEYS}"
            )
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {self.scenario}; expected 1-4")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.stakeholders < 1:
            raise ValueError("at least one stakeholder is required")
        if self.max_distance_km <= 0.0:
            raise ValueError("max_distance_km must be positive")
        self.qolu_reward_config()  # validates the reward reading

    @property
    def grid_size(self) -> int:
        return self.scale.grid_size

    @property
    def grid_shape(self) -> tuple[int, int]:
        """``(rows, columns)`` of this run's parcel grid."""
        return self.scale.shape

    @property
    def conversion_fraction(self) -> float:
        return self.scale.conversion_fraction

    @property
    def max_conversions(self) -> int | None:
        return self.scale.max_conversions

    @property
    def config_label(self) -> str:
        return self.scale.label

    def env_config(self) -> EnvConfig:
        """The shared environment settings of this run's cell."""
        return EnvConfig(
            grid_size=self.scale.grid_height,
            grid_width=self.scale.width,
            iterations=self.iterations,
            conversion_fraction=self.scale.conversion_fraction,
            max_conversions_per_iteration=self.scale.max_conversions,
        )

    def qolu_reward_config(self) -> QOLURewardConfig:
        """The reward reading and weighting switches this run is driven by."""
        return QOLURewardConfig(
            mode=self.qolu_reward_mode,
            scale_by_scenario_weight=self.scale_by_scenario_weight,
            weighted_consensus=self.weighted_consensus,
        )


def grid_digest(grid: ParcelGrid) -> str:
    """Short content hash of a parcel grid, for cross-method crop checks."""
    payload = np.ascontiguousarray(grid.composition, dtype=np.float32).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _build_baseline(env: UrbanLandUseEnv, spec: RunSpec) -> BaselineMethod:
    """Construct the baseline method named by ``spec``.

    Design note: P-MADDPG runs one agent per top-down stakeholder, so
    ``--stakeholders`` drives ``PMADDPGConfig.n_agents`` too; otherwise a
    campaign that changed the stakeholder count would silently compare three
    P-MADDPG agents against a different number of QOLU agents.

    Design note: P-MADDPG is handed the *same* :class:`QOLURewardConfig` as
    QOLU + BUIA - reward reading and both weighting switches - because it
    learns from the same signal and votes with the same consensus rule.  Only
    the published system has an entry point of its own; the other four
    baselines derive their rewards differently and take nothing from here.

    Design note (P-MADDPG variants): the three readings of the P-MADDPG actor
    are resolved here, by the same mechanism that resolves ``fixed_ltos`` and
    ``mitm_learned`` - a method *key* of its own, so they can be run side by
    side and can never overwrite each other's records.  The key is translated
    into the ``variant`` name ``PMADDPGMethod.for_env`` takes; see
    :data:`chapter4_qolu.methods.PMADDPG_METHOD_VARIANTS`.
    """
    variant = PMADDPG_METHOD_VARIANTS.get(spec.method)
    if variant is not None:
        overrides: dict[str, Any] = {
            "n_agents": spec.stakeholders,
            "qolu_reward": spec.qolu_reward_config(),
        }
        if _pmaddpg_variants_available():
            overrides["variant"] = variant
        elif variant != "published":
            raise ValueError(
                f"method {spec.method!r} needs the revised P-MADDPG "
                "(baselines.pmaddpg.PMADDPG_VARIANTS); the copy on this "
                "checkout only has the first reconstruction, so only "
                "'pmaddpg' can be run"
            )
        return PMADDPGMethod.for_env(env, seed=spec.seed, **overrides)
    if spec.method == "ltos":
        return LToSMethod.for_env(env, seed=spec.seed)
    if spec.method == "fixed_ltos":
        return fixed_ltos(env, seed=spec.seed)
    if spec.method == "mitm":
        return MitMMethod.for_env(env, seed=spec.seed)
    if spec.method == "mitm_learned":
        return learned_mitm(env, seed=spec.seed)
    if spec.method == NO_REGULATOR_METHOD:
        return NoRegulatorMethod.for_env(env, seed=spec.seed)
    raise ValueError(f"{spec.method!r} is not a baseline method")


def _pmaddpg_variants_available() -> bool:
    """Does the P-MADDPG module provide its ``variant`` switch?

    A feature check rather than a version check: the module exports
    ``PMADDPG_VARIANTS`` when it does, so its presence is exactly the question
    being asked.
    """
    try:
        from chapter4_qolu.baselines.pmaddpg import PMADDPG_VARIANTS  # noqa: F401
    except ImportError:
        return False
    return True


def _window_transform(raster: Any, origin: tuple[int, int]) -> Any | None:
    """Affine transform of the *window*, for rasterising a region boundary.

    ``None`` unless rasterio is installed and the raster is georeferenced -
    which is exactly when a boundary could be projected onto it anyway.
    """
    if raster.origin_xy is None:
        return None
    try:
        from affine import Affine  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - depends on the interpreter
        return None
    pixel = float(raster.pixel_size_m)
    east = float(raster.origin_xy[0]) + origin[1] * pixel
    north = float(raster.origin_xy[1]) - origin[0] * pixel
    return Affine(pixel, 0.0, east, 0.0, -pixel, north)


def qolu_config_for(
    method: str,
    observation_channels: int,
    window: int,
    overrides: Mapping[str, Any] | None = None,
) -> QOLUConfig:
    """The :class:`QOLUConfig` the QOLU method *key* names.

    Design note (QOLU variants): resolved here, by the same mechanism that
    resolves the P-MADDPG and baseline variants - a method *key* of its own, so
    the three readings can be run side by side and can never overwrite each
    other's records.  See :data:`chapter4_qolu.methods.QOLU_METHOD_VARIANTS`.

    ``overrides`` moves named :class:`QOLUConfig` fields off their defaults
    *after* the variant has been resolved, which is the one route by which a
    single run's learner settings may differ from the package's.  It exists
    because some of those settings are free choices of the implementation -
    ``target_sync_interval`` is the clearest case, since Algorithm 1 says only
    "every ``t_n`` steps reset ``theta^- = theta``" and Table 1 never gives
    ``t_n`` - and choosing between them needs an experiment rather than an
    edit to :mod:`chapter4_qolu.config`.  Applying them here keeps the variant
    resolution, the validation in ``QOLUConfig.__post_init__`` and the
    construction order shared between every caller; see
    :mod:`chapter4_qolu.hparam_worker` and
    :mod:`chapter4_qolu.run_distribution_diagnostics`.
    """
    try:
        variant = QOLU_METHOD_VARIANTS[str(method)]
    except KeyError:
        raise ValueError(
            f"{method!r} is not a QOLU method; expected one of "
            f"{tuple(QOLU_METHOD_VARIANTS)}"
        ) from None
    config = QOLUConfig(
        observation_channels=observation_channels,
        observation_window=window,
        **qolu_variant_settings(variant),  # type: ignore[arg-type]
    )
    if not overrides:
        return config
    known = {field.name for field in fields(QOLUConfig)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(
            f"QOLUConfig has no such setting(s) to override: {unknown}; "
            f"expected names from {sorted(known)}"
        )
    return replace(config, **dict(overrides))


def _run_joint(
    env: UrbanLandUseEnv, spec: RunSpec
) -> tuple[JointTrainingReport, dict[str, Any]]:
    """The published system: BUIA proposes, the QOLU stakeholders appraise.

    Returns the report and the run's provenance block (the resolved configs
    and what the learners did), read off the agents once training is over.
    """
    env_config = env.config
    config = qolu_config_for(
        spec.method,
        observation_channels=env.observation_channels,
        window=env_config.observation_window,
    )
    stakeholders = [
        QOLUAgent(config, seed=spec.seed + index) for index in range(spec.stakeholders)
    ]
    investor = BUIAAgent(
        BUIAConfig(
            observation_channels=env.buia_observation_channels,
            observation_window=env_config.buia_observation_window,
            max_proposals=buia_proposal_limit(env_config),
        ),
        seed=spec.seed,
    )
    report = train_joint(
        env,
        stakeholders,
        investor,
        iterations=spec.iterations,
        seed=spec.seed,
        qolu_reward=spec.qolu_reward_config(),
    )
    return report, learner_provenance(stakeholders, investor)


def learner_provenance(
    stakeholders: Sequence[QOLUAgent], investor: BUIAAgent | None
) -> dict[str, Any]:
    """The ``provenance`` block of a run record, from the agents that ran."""
    agents = list(stakeholders)
    return {
        "qolu_config": config_record(agents[0].config) if agents else None,
        "learner": learner_record(agents) if agents else None,
        "investor_config": (
            config_record(investor.config) if investor is not None else None
        ),
    }


def execute_run(spec: RunSpec) -> dict[str, Any]:
    """Run one campaign cell and write its JSON record."""
    torch.set_num_threads(max(1, int(spec.torch_threads)))

    setup_started = time.perf_counter()
    raster = load_lcm_raster(spec.raster)
    origin = resolve_window_origin(
        raster,
        spec.grid_shape,
        origin=spec.window_origin,
        bng_origin_km=spec.window_bng_km,
    )
    grid = select_region(raster, shape=spec.grid_shape, origin=origin)
    grid = apply_region_mask(
        grid,
        load_region_mask(
            spec.region_mask,
            shape=spec.grid_shape,
            transform=_window_transform(raster, origin),
        ),
    )
    env_config = spec.env_config()
    env = UrbanLandUseEnv(grid, config=env_config, weights=SCENARIOS[spec.scenario])
    digest = grid_digest(env.initial_grid)
    setup = time.perf_counter() - setup_started

    started = time.perf_counter()
    if spec.method in QOLU_METHOD_VARIANTS:
        report: JointTrainingReport | BaselineReport
        report, provenance = _run_joint(env, spec)
    else:
        method = _build_baseline(env, spec)
        report = run_baseline(
            env,
            method,
            iterations=spec.iterations,
            seed=spec.seed,
        )
        investor = getattr(method, "investor", None)
        provenance = learner_provenance(
            (), investor if isinstance(investor, BUIAAgent) else None
        )
    runtime = time.perf_counter() - started

    record = run_record(
        spec,
        raster,
        env,
        report,
        runtime,
        setup,
        digest,
        origin,
        default_method_name=spec.method,
        provenance=provenance,
    )
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    spec.output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
