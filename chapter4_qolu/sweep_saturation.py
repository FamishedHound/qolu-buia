"""Saturation / sensitivity sweep for the published iteration protocol.

At the published scale the 50 x 50 landscape can fully urbanise well before
the published 1000 iterations are up.  This sweep varies the three
things that control how fast land can be consumed - the extent
(``grid_size``), how much of a parcel one conversion takes
(``conversion_fraction``) and which reading of the reward allocation the
stakeholders learn from (``qolu_reward_mode``) - and records when, or whether,
the landscape saturates.

Usage::

    python -m chapter4_qolu.sweep_saturation --raster path/to/ukceh_download.zip

Outputs ``results/saturation_sweep_seed<seed>.json`` and a matching ``.md``
table.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
    LCMRaster,
    load_lcm_raster,
    select_region_of_interest,
)
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.experiment import DEFAULT_RESULTS_DIR, resolve_raster
from chapter4_qolu.runner import JointTrainingReport, train_joint

__all__ = [
    "RunOutcome",
    "SweepSettings",
    "first_saturated_iteration",
    "main",
    "render_markdown_table",
    "run_sweep",
    "sweep_combinations",
]

logger = logging.getLogger(__name__)

#: Urban cover share above which the landscape counts as saturated.
SATURATION_THRESHOLD = 0.99
#: Iteration that splits the "still exploring" phase from the learned phase;
#: epsilon reaches its floor at 10_000 / 50 candidates = 200 iterations.
EXPLORATION_ITERATIONS = 200


@dataclass(frozen=True)
class SweepSettings:
    """Which runs the sweep performs."""

    raster: Path
    scenario: int = 1
    iterations: int = 1000
    seed: int = 0
    stakeholders: int = 3
    grid_sizes: tuple[int, ...] = (50, 100)
    conversion_fractions: tuple[float, ...] = (1.0, 0.25)
    reward_modes: tuple[str, ...] = ("team", "per_parcel")
    output_dir: Path = DEFAULT_RESULTS_DIR

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {self.scenario}; expected 1-4")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if not self.grid_sizes:
            raise ValueError("at least one grid size is required")
        if not self.conversion_fractions:
            raise ValueError("at least one conversion fraction is required")
        if not self.reward_modes:
            raise ValueError("at least one reward mode is required")
        for mode in self.reward_modes:
            QOLURewardConfig(mode=mode)  # validates the mode


@dataclass(frozen=True)
class RunOutcome:
    """What one sweep cell produced."""

    grid_size: int
    conversion_fraction: float
    reward_mode: str
    saturation_iteration: int | None
    approval_rate_early: float
    approval_rate_late: float
    applied_rate_early: float
    applied_rate_late: float
    total_conversions: int
    runtime_seconds: float
    land_fraction: float
    metrics: Mapping[str, float] = field(default_factory=dict)
    phase_counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def as_record(self) -> dict[str, object]:
        return {
            "grid_size": self.grid_size,
            "conversion_fraction": self.conversion_fraction,
            "reward_mode": self.reward_mode,
            "saturation_iteration": self.saturation_iteration,
            "approval_rate_early": round(self.approval_rate_early, 4),
            "approval_rate_late": round(self.approval_rate_late, 4),
            "applied_rate_early": round(self.applied_rate_early, 4),
            "applied_rate_late": round(self.applied_rate_late, 4),
            "total_conversions": self.total_conversions,
            "runtime_seconds": round(self.runtime_seconds, 1),
            "land_fraction": round(self.land_fraction, 4),
            "metrics": {key: round(value, 4) for key, value in self.metrics.items()},
            "phase_counts": {
                phase: dict(counts) for phase, counts in self.phase_counts.items()
            },
        }


def sweep_combinations(
    grid_sizes: Sequence[int],
    conversion_fractions: Sequence[float],
    reward_modes: Sequence[str],
) -> tuple[tuple[int, float, str], ...]:
    """Every (grid size, conversion fraction, reward mode) cell of the sweep."""
    return tuple(
        itertools.product(
            tuple(grid_sizes), tuple(conversion_fractions), tuple(reward_modes)
        )
    )


def first_saturated_iteration(
    urban_fraction: Sequence[float], threshold: float = SATURATION_THRESHOLD
) -> int | None:
    """Index of the first iteration whose urban cover exceeds ``threshold``."""
    for iteration, value in enumerate(urban_fraction):
        if value > threshold:
            return iteration
    return None


def _phase_counts(report: JointTrainingReport, start: int, stop: int) -> dict[str, int]:
    """Total candidates / approvals / proposals / conversions over a window."""
    return {
        "candidates": int(sum(report.n_candidates[start:stop])),
        "approved": int(sum(report.n_approved[start:stop])),
        "proposals": int(sum(report.n_proposals[start:stop])),
        "conversions": int(sum(report.n_conversions[start:stop])),
    }


def _approval_rate(report: JointTrainingReport, start: int, stop: int) -> float:
    """Share of appraised candidates the stakeholders approved.

    Design note: the headline "approval rate" is measured at *appraisal* level
    (``n_approved / n_candidates``) rather than at conversion level.  Once a
    landscape saturates, no proposal can be executed any more, so a
    converted-over-proposed rate silently drops to zero and would be read as
    "the agents learnt to refuse" when in fact there was nothing left to
    convert.  :func:`_applied_rate` reports the conversion-level number
    separately, and the phase counts are stored in the JSON so either can be
    recomputed.
    """
    counts = _phase_counts(report, start, stop)
    if counts["candidates"] <= 0:
        return 0.0
    return counts["approved"] / counts["candidates"]


def _applied_rate(report: JointTrainingReport, start: int, stop: int) -> float:
    """Share of proposed parcels that were actually converted in a window."""
    counts = _phase_counts(report, start, stop)
    if counts["proposals"] <= 0:
        return 0.0
    return counts["conversions"] / counts["proposals"]


def _run_cell(
    raster: LCMRaster,
    settings: SweepSettings,
    grid_size: int,
    conversion_fraction: float,
    reward_mode: str,
) -> RunOutcome:
    grid = select_region_of_interest(raster, grid_size=grid_size)
    env_config = EnvConfig(
        grid_size=grid_size,
        iterations=settings.iterations,
        conversion_fraction=conversion_fraction,
    )
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
        qolu_reward=QOLURewardConfig(mode=reward_mode),
    )
    runtime = time.perf_counter() - started

    split = min(EXPLORATION_ITERATIONS, settings.iterations)
    return RunOutcome(
        grid_size=grid_size,
        conversion_fraction=conversion_fraction,
        reward_mode=reward_mode,
        saturation_iteration=first_saturated_iteration(report.urban_fraction),
        approval_rate_early=_approval_rate(report, 0, split),
        approval_rate_late=_approval_rate(report, split, settings.iterations),
        applied_rate_early=_applied_rate(report, 0, split),
        applied_rate_late=_applied_rate(report, split, settings.iterations),
        total_conversions=int(sum(report.n_conversions)),
        runtime_seconds=runtime,
        land_fraction=float(grid.land_mask().mean()),
        metrics=dict(report.final_metrics),
        phase_counts={
            "early": _phase_counts(report, 0, split),
            "late": _phase_counts(report, split, settings.iterations),
        },
    )


def run_sweep(settings: SweepSettings) -> list[RunOutcome]:
    """Run every sweep cell and write the JSON + markdown records."""
    raster = load_lcm_raster(settings.raster)
    outcomes: list[RunOutcome] = []
    skipped: list[dict[str, object]] = []

    for grid_size, conversion_fraction, reward_mode in sweep_combinations(
        settings.grid_sizes, settings.conversion_fractions, settings.reward_modes
    ):
        if grid_size > raster.height or grid_size > raster.width:
            reason = (
                f"raster is {raster.height}x{raster.width} parcels, too small for a "
                f"{grid_size}x{grid_size} crop"
            )
            if not any(entry["grid_size"] == grid_size for entry in skipped):
                logger.warning("skipping grid_size=%d: %s", grid_size, reason)
            skipped.append(
                {
                    "grid_size": grid_size,
                    "conversion_fraction": conversion_fraction,
                    "reward_mode": reward_mode,
                    "reason": reason,
                }
            )
            continue

        logger.info(
            "running grid=%d fraction=%.2f mode=%s",
            grid_size,
            conversion_fraction,
            reward_mode,
        )
        outcome = _run_cell(
            raster, settings, grid_size, conversion_fraction, reward_mode
        )
        logger.info(
            "  saturated at %s, %d conversions, %.0f s",
            outcome.saturation_iteration,
            outcome.total_conversions,
            outcome.runtime_seconds,
        )
        outcomes.append(outcome)

    _write_outputs(settings, raster, outcomes, skipped)
    return outcomes


def _write_outputs(
    settings: SweepSettings,
    raster: LCMRaster,
    outcomes: Sequence[RunOutcome],
    skipped: Sequence[Mapping[str, object]],
) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"saturation_sweep_seed{settings.seed}"

    record = {
        "scenario": settings.scenario,
        "iterations": settings.iterations,
        "seed": settings.seed,
        "stakeholders": settings.stakeholders,
        "raster": {
            "path": str(raster.path),
            "height": raster.height,
            "width": raster.width,
        },
        "runs": [outcome.as_record() for outcome in outcomes],
        "skipped": list(skipped),
    }
    (settings.output_dir / f"{stem}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    (settings.output_dir / f"{stem}.md").write_text(
        _markdown_document(settings, outcomes, skipped), encoding="utf-8"
    )


def render_markdown_table(outcomes: Sequence[RunOutcome]) -> str:
    """One row per sweep cell."""
    columns = (
        "grid",
        "conv. frac",
        "reward mode",
        "saturated at",
        "approve 0-200",
        "approve 200+",
        "applied 200+",
        "conversions",
        "land_use",
        "woodland",
        "urban",
        "C_align",
        "WB",
        "runtime s",
    )
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "---|" * len(columns)
    rows = []
    for outcome in outcomes:
        metrics = outcome.metrics
        saturated = (
            "never"
            if outcome.saturation_iteration is None
            else str(outcome.saturation_iteration)
        )
        rows.append(
            "| {grid}x{grid} | {fraction:.2f} | {mode} | {saturated} | {early:.2f} | "
            "{late:.2f} | {applied:.2f} | {conversions} | {land_use:.3f} | "
            "{woodland:.3f} | {urban:.3f} | {align:.4f} | {well_being:.3f} | "
            "{runtime:.0f} |".format(
                grid=outcome.grid_size,
                fraction=outcome.conversion_fraction,
                mode=outcome.reward_mode,
                saturated=saturated,
                early=outcome.approval_rate_early,
                late=outcome.approval_rate_late,
                applied=outcome.applied_rate_late,
                conversions=outcome.total_conversions,
                land_use=metrics.get("land_use", float("nan")),
                woodland=metrics.get("woodland", float("nan")),
                urban=metrics.get("urban", float("nan")),
                align=metrics.get("conversion_alignment", float("nan")),
                well_being=metrics.get("well_being", float("nan")),
                runtime=outcome.runtime_seconds,
            )
        )
    return "\n".join([header, separator, *rows])


def _markdown_document(
    settings: SweepSettings,
    outcomes: Sequence[RunOutcome],
    skipped: Sequence[Mapping[str, object]],
) -> str:
    setup = (
        f"{settings.iterations} iterations per run, {settings.stakeholders} QOLU "
        f"stakeholders + BUIA, on `{settings.raster.name}`."
    )
    legend = (
        "`saturated at` is the first iteration whose urban cover exceeds "
        f"{SATURATION_THRESHOLD:.2f} (`never` if it does not). `approve` columns are "
        "approved / appraised candidate parcels before and after epsilon reaches its "
        f"floor at iteration {EXPLORATION_ITERATIONS}; `applied 200+` is converted / "
        "proposed parcels over the same late window, which necessarily falls to zero "
        "once a landscape has saturated."
    )
    lines = [
        f"# Saturation sweep (scenario {settings.scenario}, seed {settings.seed})",
        "",
        setup,
        "",
        legend,
        "",
        render_markdown_table(outcomes),
        "",
    ]
    if skipped:
        lines.append("## Skipped cells")
        lines.append("")
        for entry in skipped:
            lines.append(
                f"* grid {entry['grid_size']}, fraction {entry['conversion_fraction']}, "
                f"mode {entry['reward_mode']}: {entry['reason']}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", default=None, help="LCM GeoTIFF or UKCEH zip")
    parser.add_argument("--scenario", type=int, default=1, choices=sorted(SCENARIOS))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stakeholders", type=int, default=3)
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = SweepSettings(
        raster=resolve_raster(arguments.raster),
        scenario=arguments.scenario,
        iterations=arguments.iterations,
        seed=arguments.seed,
        stakeholders=arguments.stakeholders,
        output_dir=Path(arguments.output_dir),
    )
    outcomes = run_sweep(settings)
    logger.info("\n%s", render_markdown_table(outcomes))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
