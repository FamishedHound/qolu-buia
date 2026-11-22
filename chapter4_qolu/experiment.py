"""End-to-end joint QOLU + BUIA run on a UKCEH LCM raster.

This is a single-scenario driver; the paper's four-scenario experiment runner is
:mod:`chapter4_qolu.run_scenarios`.  It loads a raster, builds the environment
for one preference setting, trains the investor and the stakeholders jointly for
the published 1000 iterations and writes a JSON record of the run.

Usage::

    python -m chapter4_qolu.experiment --raster path/to/lcm.tif --scenario 1
    python -m chapter4_qolu.experiment --raster path/to/ukceh_download.zip

    # let the scenario's preference weights reach the stakeholders themselves
    # (both switches are off by default; see config.QOLURewardConfig)
    python -m chapter4_qolu.experiment --raster path/to/lcm.tif --scenario 3 \\
        --scale-rewards-by-weight --weighted-consensus

The raster path may also come from ``CHAPTER4_LCM_TIF``; a ``.zip`` is unpacked
to a temporary directory first, so the UKCEH download works as-is.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from chapter4_qolu.agent.buia_agent import BUIAAgent
from chapter4_qolu.agent.qolu_agent import QOLUAgent
from chapter4_qolu.config import (
    SCENARIOS,
    BUIAConfig,
    EnvConfig,
    QOLUConfig,
    QOLURewardConfig,
)
from chapter4_qolu.data.lcm_loader import (
    is_placeholder_region,
    load_lcm_raster,
    select_region_of_interest,
)
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.runner import JointTrainingReport, train_joint

__all__ = ["ExperimentSettings", "main", "resolve_raster", "run_joint_scenario"]

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class ExperimentSettings:
    """Everything that defines one joint run."""

    raster: Path
    scenario: int = 1
    iterations: int = 1000
    grid_size: int = 50
    #: The paper evaluates three top-down stakeholders, one per preference.
    stakeholders: int = 3
    seed: int = 0
    #: Which reading of the paper's reward allocation to train on; defaults to
    #: Section 3.5's per-parcel compliance signal.
    qolu_reward_mode: str = "per_parcel"
    #: Optional routes by which this scenario's weights reach the stakeholders
    #: (see :class:`~chapter4_qolu.config.QOLURewardConfig`).  Both off by
    #: default: with the published reading the weights change nothing but the
    #: reported team reward, which is why every scenario currently produces the
    #: same trajectory for a given seed.
    scale_by_scenario_weight: bool = False
    weighted_consensus: bool = False
    output_dir: Path = DEFAULT_RESULTS_DIR

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {self.scenario}; expected 1-4")
        self.qolu_reward_config()  # validates the reward reading
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.stakeholders < 1:
            raise ValueError("at least one stakeholder is required")

    def qolu_reward_config(self) -> QOLURewardConfig:
        """The reward reading and weighting switches this run is driven by."""
        return QOLURewardConfig(
            mode=self.qolu_reward_mode,
            scale_by_scenario_weight=self.scale_by_scenario_weight,
            weighted_consensus=self.weighted_consensus,
        )


def resolve_raster(path: str | Path | None) -> Path:
    """Resolve a raster path, unpacking a UKCEH download zip when needed."""
    candidate = path or os.environ.get("CHAPTER4_LCM_TIF")
    if candidate is None:
        raise ValueError(
            "no raster given: pass --raster or set CHAPTER4_LCM_TIF to an LCM "
            "percentage-aggregate-class GeoTIFF (or the UKCEH download zip)"
        )
    resolved = Path(candidate)
    if not resolved.exists():
        raise FileNotFoundError(f"raster not found: {resolved}")
    if resolved.suffix.lower() != ".zip":
        return resolved

    with zipfile.ZipFile(resolved) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".tif")]
        if not members:
            raise ValueError(f"{resolved.name} contains no GeoTIFF")
        target = Path(tempfile.mkdtemp(prefix="chapter4_lcm_"))
        archive.extract(members[0], target)
        return target / members[0]


def run_joint_scenario(settings: ExperimentSettings) -> dict[str, object]:
    """Train BUIA + QOLU jointly and write a JSON record of the run."""
    raster = load_lcm_raster(settings.raster)
    grid = select_region_of_interest(raster, grid_size=settings.grid_size)

    env_config = EnvConfig(grid_size=settings.grid_size, iterations=settings.iterations)
    env = UrbanLandUseEnv(grid, config=env_config, weights=SCENARIOS[settings.scenario])

    stakeholders = [
        QOLUAgent(
            QOLUConfig(
                observation_channels=env.observation_channels,
                observation_window=env_config.observation_window,
            ),
            seed=settings.seed + index,
        )
        for index in range(settings.stakeholders)
    ]
    investor = BUIAAgent(
        BUIAConfig(
            observation_channels=env.buia_observation_channels,
            observation_window=env_config.buia_observation_window,
            max_proposals=env_config.max_conversions_per_iteration,
        ),
        seed=settings.seed,
    )

    started = time.perf_counter()
    report = train_joint(
        env,
        stakeholders,
        investor,
        iterations=settings.iterations,
        seed=settings.seed,
        qolu_reward=settings.qolu_reward_config(),
    )
    runtime = time.perf_counter() - started

    record = _as_record(settings, raster.path, env, report, runtime)
    output_path = settings.output_dir / (
        f"scenario{settings.scenario}_joint_seed{settings.seed}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("wrote %s (%.1f s)", output_path, runtime)
    record["output_path"] = str(output_path)
    return record


def _as_record(
    settings: ExperimentSettings,
    raster_path: Path,
    env: UrbanLandUseEnv,
    report: JointTrainingReport,
    runtime: float,
) -> dict[str, object]:
    weights = SCENARIOS[settings.scenario]
    return {
        "scenario": settings.scenario,
        "weights": {
            "land_use": weights.land_use,
            "woodland": weights.woodland,
            "urban": weights.urban,
        },
        "iterations": report.iterations,
        "episodes": report.episodes,
        "seed": settings.seed,
        "grid_size": settings.grid_size,
        "stakeholders": settings.stakeholders,
        "qolu_reward_mode": settings.qolu_reward_mode,
        "scale_by_scenario_weight": bool(settings.scale_by_scenario_weight),
        "weighted_consensus": bool(settings.weighted_consensus),
        "raster": {
            "path": str(raster_path),
            "placeholder_region": is_placeholder_region(raster_path),
        },
        "runtime_seconds": round(runtime, 2),
        "final_metrics": dict(report.final_metrics),
        "summary": {
            "mean_team_reward": report.mean_team_reward,
            "mean_buia_reward": report.mean_buia_reward,
            "final_urban_fraction": report.urban_fraction[-1],
            "initial_urban_fraction": report.urban_fraction[0],
            "total_conversions": int(sum(report.n_conversions)),
            "final_well_being": report.well_being[-1],
        },
        "per_iteration": {
            "team_reward": list(report.team_rewards),
            "buia_reward": list(report.buia_rewards),
            "well_being": list(report.well_being),
            "urban_fraction": list(report.urban_fraction),
            "conversion_alignment": list(report.conversion_alignment),
            "n_conversions": list(report.n_conversions),
            "n_proposals": list(report.n_proposals),
            "n_approved": list(report.n_approved),
            "n_candidates": list(report.n_candidates),
            "epsilon": list(report.epsilons),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", default=None, help="LCM GeoTIFF or UKCEH zip")
    parser.add_argument("--scenario", type=int, default=1, choices=sorted(SCENARIOS))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--stakeholders", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--qolu-reward-mode",
        default="per_parcel",
        choices=("per_parcel", "objective", "team"),
        help="reward allocation: Section 3.5 per parcel (default), Section 3.5 "
        "per iteration, or Section 3.1's shared team reward",
    )
    parser.add_argument(
        "--scale-rewards-by-weight",
        action="store_true",
        help="pay stakeholder i n * w_i times its own per-parcel reward, i.e. "
        "the per-parcel reading of R = sum_i w_i R_phi_i (off by default)",
    )
    parser.add_argument(
        "--weighted-consensus",
        action="store_true",
        help="weight each stakeholder's contribution to the consensus sum by "
        "its scenario weight (off by default)",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = ExperimentSettings(
        raster=resolve_raster(arguments.raster),
        scenario=arguments.scenario,
        iterations=arguments.iterations,
        grid_size=arguments.grid_size,
        stakeholders=arguments.stakeholders,
        seed=arguments.seed,
        qolu_reward_mode=arguments.qolu_reward_mode,
        scale_by_scenario_weight=arguments.scale_rewards_by_weight,
        weighted_consensus=arguments.weighted_consensus,
        output_dir=Path(arguments.output_dir),
    )
    record = run_joint_scenario(settings)
    summary = record["summary"]
    assert isinstance(summary, dict)
    logger.info("final metrics: %s", record["final_metrics"])
    logger.info("summary: %s", summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
