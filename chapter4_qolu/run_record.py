"""The JSON record one campaign run writes.

Split out of :mod:`chapter4_qolu.scenario_worker` so that "what a run does"
stays separate from "what a run leaves behind".  Everything a later reader
needs in order to know what produced a number lives here: the method, the cell,
the seed, the *full* environment settings, both scenario-weighting switches, the
raster and the window it was cropped from, a digest of the initial landscape,
both metric views and the per-iteration series.

``initial_grid_digest`` is the one that matters for comparability: every method
inside a (configuration, scenario, seed) cell has to have seen the same
landscape, and the digest is how that is *checked* rather than assumed.

A record also carries the landscape itself, twice: ``initial_grid`` and
``final_grid``, each a dominant-class map of the same shape as the parcel
grid.  That is what makes a before/after map reproducible from the record
alone, without the raster, the window and a re-run.  Everything downstream
treats the two keys as optional, so a record without them is still read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from typing import TYPE_CHECKING, Any

from chapter4_qolu.config import SCENARIOS, EnvConfig
from chapter4_qolu.data.lcm_loader import LCMRaster, is_placeholder_region
from chapter4_qolu.env.class_grids import (
    GRID_NO_DATA,
    class_grid,
    decode_class_grid,
    encode_class_grid,
)
from chapter4_qolu.env.evaluation import evaluate_landscape
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.reporting.aggregate import saturation_iteration

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from chapter4_qolu.agent.qolu_agent import QOLUAgent
    from chapter4_qolu.baselines.base import BaselineReport
    from chapter4_qolu.runner import JointTrainingReport
    from chapter4_qolu.scenario_worker import RunSpec

__all__ = [
    "GRID_NO_DATA",
    "SERIES_FIELDS",
    "absolute_metrics",
    "class_grid",
    "config_record",
    "decode_class_grid",
    "encode_class_grid",
    "env_settings",
    "learner_record",
    "run_record",
]


#: Series every report - joint or baseline - can supply, so one record shape
#: covers all six methods.
SERIES_FIELDS: tuple[tuple[str, str], ...] = (
    ("team_reward", "team_rewards"),
    ("urban_fraction", "urban_fraction"),
    ("n_conversions", "n_conversions"),
    ("n_proposals", "n_proposals"),
    ("n_approved", "n_approved"),
    ("n_candidates", "n_candidates"),
    ("conversion_alignment", "conversion_alignment"),
    ("well_being", "well_being"),
)


def env_settings(config: EnvConfig) -> dict[str, Any]:
    """Every environment setting that has to match across a cell's methods."""
    reward = config.reward
    return {
        "grid_size": config.grid_size,
        "grid_width": config.grid_shape[1],
        "samples_per_iteration": config.samples_per_iteration,
        "max_conversions_per_iteration": config.max_conversions_per_iteration,
        "iterations": config.iterations,
        "observation_window": config.observation_window,
        "buia_observation_window": config.buia_observation_window,
        "conversion_fraction": config.conversion_fraction,
        "urban_saturation": config.urban_saturation,
        "reward": {
            "alpha": reward.alpha,
            "beta": reward.beta,
            "presence_threshold": reward.presence_threshold,
            "max_distance": reward.max_distance,
            "separation_group": reward.separation_group.value,
            "protected_groups": [group.value for group in reward.protected_groups],
            "societal_forest_group": reward.societal_forest_group.value,
            "societal_agricultural_group": reward.societal_agricultural_group.value,
        },
        "attractiveness": {
            "radius": config.attractiveness.radius,
            "decay_length": config.attractiveness.decay_length,
            "group_weights": [
                [group.value, weight]
                for group, weight in config.attractiveness.group_weights
            ],
        },
        "well_being": {
            "capacity_per_parcel": config.well_being.capacity_per_parcel,
            "initial_occupancy": config.well_being.initial_occupancy,
            "population_growth_per_iteration": (
                config.well_being.population_growth_per_iteration
            ),
        },
    }


def absolute_metrics(env: UrbanLandUseEnv, max_distance_km: float) -> dict[str, float]:
    """The five metrics with distances saturating at a physical distance.

    Parcels are 1 km squares, so ``RewardConfig.max_distance`` is already in
    kilometres; the default (``None``) is the grid diagonal, which is why the
    RAW ``land_use`` column sits near 0.05 for every method on this landscape.
    """
    return evaluate_landscape(
        env.initial_grid,
        env.grid,
        attractiveness=env.attractiveness,
        population=env.population,
        reward_config=replace(env.config.reward, max_distance=float(max_distance_km)),
        well_being_config=env.config.well_being,
    )


def _window_record(spec: RunSpec, origin: tuple[int, int]) -> dict[str, Any]:
    """The crop this run actually used, pinned or chosen automatically."""
    return {
        "label": spec.window_label,
        "bng_top_left_km": (
            list(spec.window_bng_km) if spec.window_bng_km is not None else None
        ),
        "raster_origin": [int(origin[0]), int(origin[1])],
        "pinned": spec.window_origin is not None or spec.window_bng_km is not None,
    }


def learner_record(stakeholders: Sequence[QOLUAgent]) -> dict[str, Any]:
    """The target-network bookkeeping of a set of stakeholders.

    Reported per stakeholder as well as in total because the stakeholders learn
    independently: three agents on the same schedule should agree, and a
    summary that only printed the total could not show that they did.

    ``target_network_ever_synced`` is the one a reader has to see.  When it is
    ``False`` every bootstrap in the run was evaluated on the target network's
    *initial*, random weights, which is a statement about the run and not about
    the algorithm.
    """
    agents = list(stakeholders)
    if not agents:
        raise ValueError("a learner record needs at least one stakeholder")
    intervals = {int(agent.config.target_sync_interval) for agent in agents}
    if len(intervals) != 1:
        raise ValueError(
            f"stakeholders disagree on target_sync_interval: {sorted(intervals)}"
        )
    budgets = {int(agent.config.learn_steps_per_iteration) for agent in agents}
    if len(budgets) != 1:
        raise ValueError(
            f"stakeholders disagree on learn_steps_per_iteration: {sorted(budgets)}"
        )
    supports = {
        (
            int(agent.support.n_atoms),
            float(agent.support.v_min),
            float(agent.support.v_max),
        )
        for agent in agents
    }
    if len(supports) != 1:
        raise ValueError(
            f"stakeholders disagree on the atom support: {sorted(supports)}"
        )
    n_atoms, v_min, v_max = supports.pop()
    syncs = [int(agent.target_syncs) for agent in agents]
    gradient_steps = [int(agent.gradient_steps) for agent in agents]
    return {
        "target_sync_interval": intervals.pop(),
        "gradient_steps": gradient_steps,
        "target_syncs": syncs,
        "total_target_syncs": sum(syncs),
        "target_network_ever_synced": sum(syncs) > 0,
        "learning_starts": int(agents[0].config.learning_starts),
        "batch_size": int(agents[0].config.batch_size),
        # The gradient budget and the atom support, stated as what the agents
        # were actually built with rather than as what the command line asked
        # for.  ``gradient_steps`` alone cannot separate "more iterations" from
        # "more gradient steps per iteration", and every distributional
        # statistic in this document is meaningless without the atom spacing.
        "learn_steps_per_iteration": budgets.pop(),
        "support": {
            "n_atoms": n_atoms,
            "v_min": v_min,
            "v_max": v_max,
            "delta_z": (v_max - v_min) / (n_atoms - 1),
        },
    }


def config_record(config: Any) -> dict[str, Any] | None:
    """A frozen config dataclass as plain JSON values, or ``None`` for none."""
    if config is None:
        return None
    if not is_dataclass(config):
        raise TypeError(f"expected a config dataclass, got {type(config).__name__}")
    return json.loads(json.dumps(asdict(config)))


def run_record(
    spec: RunSpec,
    raster: LCMRaster,
    env: UrbanLandUseEnv,
    report: JointTrainingReport | BaselineReport,
    runtime: float,
    setup: float,
    digest: str,
    origin: tuple[int, int],
    default_method_name: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything one finished run is worth keeping, as a JSON-ready dict.

    ``provenance`` names the learners that produced the run: ``qolu_config``
    (the resolved :class:`~chapter4_qolu.config.QOLUConfig`, which states
    ``t_n`` as ``target_sync_interval``), ``learner`` (what the stakeholders'
    learners actually did - see :func:`learner_record`) and
    ``investor_config`` (the BUIA config).  A method without the component
    records ``None`` under its key, so an absent learner is stated, not implied.
    """
    stated = dict(provenance or {})
    weights = SCENARIOS[spec.scenario]
    series = {
        name: list(getattr(report, attribute)) for name, attribute in SERIES_FIELDS
    }
    return {
        "method": spec.method,
        "method_name": getattr(report, "method", default_method_name),
        "scenario": spec.scenario,
        "weights": {
            "land_use": weights.land_use,
            "woodland": weights.woodland,
            "urban": weights.urban,
        },
        "seed": spec.seed,
        "iterations": report.iterations,
        "episodes": report.episodes,
        "stakeholders": spec.stakeholders,
        "qolu_reward_mode": spec.qolu_reward_mode,
        "scale_by_scenario_weight": bool(spec.scale_by_scenario_weight),
        "weighted_consensus": bool(spec.weighted_consensus),
        "qolu_config": stated.get("qolu_config"),
        "learner": stated.get("learner"),
        "investor_config": stated.get("investor_config"),
        "configuration": spec.scale.as_record(),
        "env_settings": env_settings(env.config),
        "raster": {
            "path": str(raster.path),
            "placeholder_region": is_placeholder_region(raster.path),
        },
        "window": _window_record(spec, origin),
        "initial_grid_digest": digest,
        "initial_grid": encode_class_grid(class_grid(env.initial_grid)),
        "final_grid": encode_class_grid(class_grid(env.grid)),
        "absolute_max_distance_km": float(spec.max_distance_km),
        "runtime_seconds": round(runtime, 3),
        "setup_seconds": round(setup, 3),
        "total_conversions": int(sum(report.n_conversions)),
        "saturation_iteration": saturation_iteration(report.urban_fraction),
        "final_metrics": dict(report.final_metrics),
        "final_metrics_absolute": absolute_metrics(env, spec.max_distance_km),
        "per_iteration": series,
    }
