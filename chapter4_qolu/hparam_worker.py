"""Executing one run under a perturbed hyper-parameter.

:class:`~chapter4_qolu.scenario_worker.RunSpec` fixes the *cell* a run belongs
to - method, scenario, seed, extent, cap, reward reading - and
:func:`~chapter4_qolu.scenario_worker.execute_run` performs it.  Neither knows
about hyper-parameters.
:class:`HParamRunSpec` adds exactly one field, the :class:`Perturbation` in
force, and is otherwise the same spec: the same record layout, the same result
paths, the same ``--resume`` logic.

Two kinds of perturbation reach the run by two different routes, and the split
is what keeps this module small:

* an **environment**-side one (the sampled-parcel count, the observation
  window) is expressed by overriding :meth:`RunSpec.env_config`, so
  ``execute_run`` performs the run completely unchanged and every method in the
  cell moves together;
* an **agent**-side one is a field of ``QOLUConfig``, which ``execute_run``
  builds internally from the method key alone.  That single step is re-done
  here - the config is still built by
  :func:`~chapter4_qolu.scenario_worker.qolu_config_for`, which applies the
  override itself, so the variant resolution, the validation, the BUIA
  proposal limit, the record layout and the metrics all stay shared.  That
  resolver is the *only* route by which a run's learner settings may leave
  their defaults; :mod:`chapter4_qolu.run_distribution_diagnostics` uses the
  same one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import time
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from chapter4_qolu.agent.buia_agent import BUIAAgent
from chapter4_qolu.agent.qolu_agent import QOLUAgent
from chapter4_qolu.campaign_plan import estimate_runtime_seconds
from chapter4_qolu.config import SCENARIOS, BUIAConfig, EnvConfig
from chapter4_qolu.data.lcm_loader import (
    load_lcm_raster,
    resolve_window_origin,
    select_region,
)
from chapter4_qolu.data.region_mask import apply_region_mask, load_region_mask
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.hparam_sensitivity import Perturbation
from chapter4_qolu.methods import QOLU_METHOD_VARIANTS
from chapter4_qolu.run_record import run_record
from chapter4_qolu.runner import train_joint
from chapter4_qolu.scenario_worker import (
    RunSpec,
    _window_transform,
    buia_proposal_limit,
    execute_run,
    grid_digest,
    learner_provenance,
    qolu_config_for,
)

__all__ = [
    "HParamRunSpec",
    "as_hparam_spec",
    "execute_hparam_run",
    "execute_hparam_specs",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HParamRunSpec(RunSpec):
    """A campaign run with at most one hyper-parameter moved off its default."""

    #: ``None`` for the baseline configuration's own runs.
    perturbation: Perturbation | None = None

    @property
    def env_overrides(self) -> dict[str, float | int]:
        """``EnvConfig`` fields this run moves (empty unless env-side)."""
        if self.perturbation is None or self.perturbation.target != "environment":
            return {}
        return self.perturbation.overrides()

    @property
    def qolu_overrides(self) -> dict[str, float | int]:
        """``QOLUConfig`` fields this run moves (empty unless agent-side)."""
        if self.perturbation is None or self.perturbation.target != "qolu":
            return {}
        return self.perturbation.overrides()

    def env_config(self) -> EnvConfig:
        """The cell's environment settings, with any env-side move applied."""
        config = super().env_config()
        overrides = self.env_overrides
        return replace(config, **overrides) if overrides else config

    def perturbation_record(self) -> dict[str, Any] | None:
        return None if self.perturbation is None else self.perturbation.as_record()


def as_hparam_spec(spec: RunSpec, perturbation: Perturbation | None) -> HParamRunSpec:
    """Re-label an ordinary :class:`RunSpec` as one carrying a perturbation.

    Design note: the plan is still built by
    :func:`chapter4_qolu.campaign_plan.campaign_runs`, so the result paths, the
    seeds and the window pinning are the campaign machinery's rather than a
    second copy of it; this only widens the resulting specs.
    """
    values = {field.name: getattr(spec, field.name) for field in fields(RunSpec)}
    return HParamRunSpec(**values, perturbation=perturbation)


def _perturbed_qolu_run(spec: HParamRunSpec) -> dict[str, Any]:
    """One QOLU + BUIA run whose learner config is off its default.

    The body mirrors :func:`chapter4_qolu.scenario_worker.execute_run` - it has
    to, the two must produce byte-comparable records - and differs in one line:
    the :class:`~chapter4_qolu.config.QOLUConfig` is perturbed after
    ``qolu_config_for`` has resolved the method variant.

    Design note (why ``scenario_worker._window_transform`` is imported rather
    than re-derived): ``--region-mask`` has to be projected onto the window
    with the *same* affine transform ``execute_run`` uses, and
    :func:`chapter4_qolu.data.region_mask.load_region_mask` refuses to
    rasterise a boundary without one.  Two copies of that derivation could
    drift apart and clip two runs of the same cell to different boundaries, so
    the existing one is reused as it stands.
    """
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

    config = qolu_config_for(
        spec.method,
        observation_channels=env.observation_channels,
        window=env_config.observation_window,
        overrides=spec.qolu_overrides,
    )
    # The construction *order* below is load-bearing and must stay exactly
    # ``scenario_worker._run_joint``'s: stakeholders first, investor second.
    # Both agents draw their weight initialisation from the process-wide torch
    # generator, so building them in the other order gives every network in the
    # run different initial weights.  That is invisible in the record and would
    # be attributed to the perturbation, because this function's output is only
    # ever read as the perturbed half of a paired comparison against a baseline
    # run produced by ``execute_run``.  ``tests/test_hparam_worker_parity.py``
    # pins it with the invariant that catches it: a perturbation whose value is
    # the package default has to reproduce the baseline run exactly.
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
    started = time.perf_counter()
    report = train_joint(
        env,
        stakeholders,
        investor,
        iterations=spec.iterations,
        seed=spec.seed,
        qolu_reward=spec.qolu_reward_config(),
    )
    runtime = time.perf_counter() - started

    return run_record(
        spec,
        raster,
        env,
        report,
        runtime,
        setup,
        digest,
        origin,
        default_method_name=spec.method,
        provenance=learner_provenance(stakeholders, investor),
    )


def _qolu_config_digest(overrides: dict[str, float | int]) -> str:
    """Short content hash of the learner overrides, for cross-run checks."""
    payload = json.dumps(overrides, sort_keys=True, default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def execute_hparam_run(spec: HParamRunSpec) -> dict[str, Any]:
    """Run one perturbed cell and write its JSON record.

    The record is the ordinary campaign record with one extra key,
    ``hyperparameter``, holding the whole :class:`Perturbation` - the requested
    value as well as the configured one - so a reader of the file alone can
    tell what was run without consulting the driver that ran it.
    """
    perturbs_agent = bool(spec.qolu_overrides) and spec.method in QOLU_METHOD_VARIANTS
    record = _perturbed_qolu_run(spec) if perturbs_agent else execute_run(spec)
    record["hyperparameter"] = spec.perturbation_record()
    record["qolu_overrides"] = dict(spec.qolu_overrides)
    record["qolu_overrides_digest"] = _qolu_config_digest(spec.qolu_overrides)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    spec.output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _worker(spec: HParamRunSpec) -> dict[str, Any]:
    """Pool entry point: run one cell, never raise into the parent."""
    try:
        execute_hparam_run(spec)
    except Exception as error:  # noqa: BLE001 - one bad cell must not stop 100
        return {"path": str(spec.output_path), "ok": False, "error": repr(error)}
    return {"path": str(spec.output_path), "ok": True, "error": None}


def execute_hparam_specs(
    specs: Sequence[HParamRunSpec], parallel: int
) -> list[dict[str, Any]]:
    """Run every spec, in-process when ``parallel`` is 1, else in a pool.

    Longest-first, the LPT heuristic
    :func:`chapter4_qolu.run_scenarios.execute_specs` uses; that executor
    cannot be reused directly because its pool entry point calls
    ``scenario_worker.execute_run``, which knows nothing of a perturbation.
    """
    ordered = sorted(specs, key=lambda spec: -estimate_runtime_seconds([spec]))
    total = len(ordered)
    outcomes: list[dict[str, Any]] = []

    def _report(index: int, outcome: dict[str, Any]) -> None:
        level = logging.INFO if outcome["ok"] else logging.ERROR
        logger.log(
            level,
            "[%d/%d] %s %s",
            index,
            total,
            "ok  " if outcome["ok"] else "FAIL",
            outcome["path"],
        )
        if not outcome["ok"]:
            logger.error("        %s", outcome["error"])

    if max(1, int(parallel)) == 1 or total <= 1:
        for index, spec in enumerate(ordered, start=1):
            outcome = _worker(spec)
            outcomes.append(outcome)
            _report(index, outcome)
        return outcomes

    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=min(int(parallel), total)) as pool:
        for index, outcome in enumerate(pool.imap_unordered(_worker, ordered), start=1):
            outcomes.append(outcome)
            _report(index, outcome)
    return outcomes
