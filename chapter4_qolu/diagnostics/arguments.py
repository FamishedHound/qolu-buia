"""The command line of :mod:`chapter4_qolu.run_distribution_diagnostics`.

Split out for the same reason :mod:`chapter4_qolu.diagnostics.plan` is: "what
the user asked for" is a separate concern from "doing it", and only the former
needs to be readable from a test.  The driver re-exports :func:`build_parser`
as ``_build_parser`` so the CLI keeps one public entry point.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from chapter4_qolu.config import SCENARIOS
from chapter4_qolu.diagnostics.plan import DiagnosticsSettings, scale_for_window
from chapter4_qolu.diagnostics.records import HEADLINE_TARGET
from chapter4_qolu.diagnostics.returns import TARGET_DESCRIPTIONS
from chapter4_qolu.experiment import resolve_raster
from chapter4_qolu.methods import QOLU_METHOD, QOLU_METHOD_VARIANTS
from chapter4_qolu.reporting.paths import parse_scale_config

__all__ = ["build_parser", "is_sweep", "settings_from"]


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Every flag the diagnostics driver takes, single-run and sweep alike."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raster", default=None, help="LCM raster (.img/.tif/.zip)")
    parser.add_argument("--tag", default=None, help="results folder suffix")
    parser.add_argument(
        "--method",
        default=QOLU_METHOD,
        choices=sorted(QOLU_METHOD_VARIANTS),
        help="which QOLU reading to diagnose",
    )
    parser.add_argument("--scenario", type=int, default=1, choices=sorted(SCENARIOS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        type=int,
        default=None,
        choices=sorted(SCENARIOS),
        help="sweep these scenarios instead of the single --scenario",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="sweep these seeds instead of the single --seed",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="worker processes for a sweep; each is pinned to one torch thread",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="GRID:FRACTION:CAP",
        help="environment scale, e.g. 50:1.0:10 (default: taken from --window)",
    )
    parser.add_argument(
        "--window", default=None, metavar="LABEL", help="e.g. E495_N140, SE_region"
    )
    parser.add_argument("--stakeholders", type=int, default=None)
    parser.add_argument(
        "--iterations", type=int, default=None, help="training iterations"
    )
    parser.add_argument(
        "--eval-iterations",
        type=int,
        default=None,
        help="length of the evaluation rollout the statistics are computed over",
    )
    parser.add_argument(
        "--qolu-reward-mode",
        default=None,
        choices=("per_parcel", "objective", "team"),
    )
    parser.add_argument("--scale-rewards-by-weight", action="store_true")
    parser.add_argument("--weighted-consensus", action="store_true")
    parser.add_argument(
        "--target-sync-interval",
        type=int,
        default=None,
        metavar="T_N",
        help=(
            "gradient steps between theta^- <- theta (Algorithm 1's t_n, which "
            "Table 4.1 never gives); omit for the package default"
        ),
    )
    parser.add_argument(
        "--n-atoms",
        type=int,
        default=None,
        metavar="N",
        help=("size of the C51 atom support; omit for Table 4.1's published 51"),
    )
    parser.add_argument(
        "--v-min",
        type=float,
        default=None,
        metavar="V",
        help="lower end of the atom support; omit for Table 4.1's published -200",
    )
    parser.add_argument(
        "--v-max",
        type=float,
        default=None,
        metavar="V",
        help="upper end of the atom support; omit for Table 4.1's published +200",
    )
    parser.add_argument(
        "--learn-steps-per-iteration",
        type=int,
        default=None,
        metavar="K",
        help=(
            "gradient steps per training iteration (the replay ratio, which "
            "the chapter never states); omit for the package default of 1"
        ),
    )
    parser.add_argument(
        "--campaign",
        default=None,
        metavar="DIR",
        help="campaign directory holding persisted stakeholders",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="DIR",
        help="directory of stakeholder<i>.pt checkpoints to diagnose",
    )
    parser.add_argument(
        "--no-investor",
        action="store_true",
        help="run the evaluation rollout without a bottom-up investor",
    )
    parser.add_argument(
        "--headline-target",
        default=HEADLINE_TARGET,
        choices=sorted(TARGET_DESCRIPTIONS),
        help="which realised-return reading the verdict is based on",
    )
    parser.add_argument(
        "--diagnostics-seed",
        type=int,
        default=0,
        help="seed of the PIT randomisation, so the KS statistic reproduces",
    )
    parser.add_argument("--decimals", type=int, default=None, choices=(3, 4, 6))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse a finished summary and any agents already persisted on disk",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild the sweep documents from the cells already on disk",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and the ETA, then stop"
    )
    parser.add_argument(
        "--out", "--output-dir", dest="out", default=None, metavar="DIR"
    )
    return parser


def _raster_for(arguments: argparse.Namespace) -> Path:
    """The raster, or a stand-in when ``--report-only`` will not open one."""
    if arguments.report_only and not arguments.raster:
        return Path("(not loaded: --report-only)")
    return resolve_raster(arguments.raster)


def settings_from(arguments: argparse.Namespace) -> DiagnosticsSettings:
    """Turn a parsed command line into the settings one run (or cell) needs."""
    defaults = DiagnosticsSettings(raster=_raster_for(arguments))
    scale = (
        parse_scale_config(arguments.config)
        if arguments.config
        else scale_for_window(arguments.window, defaults.scale)
    )
    return replace(
        defaults,
        tag=arguments.tag or defaults.tag,
        method=arguments.method,
        scenario=arguments.scenario,
        seed=arguments.seed,
        scale=scale,
        window_label=arguments.window,
        stakeholders=arguments.stakeholders or defaults.stakeholders,
        iterations=arguments.iterations or defaults.iterations,
        eval_iterations=arguments.eval_iterations or defaults.eval_iterations,
        qolu_reward_mode=arguments.qolu_reward_mode or defaults.qolu_reward_mode,
        scale_by_scenario_weight=arguments.scale_rewards_by_weight,
        weighted_consensus=arguments.weighted_consensus,
        target_sync_interval=arguments.target_sync_interval,
        n_atoms=arguments.n_atoms,
        v_min=arguments.v_min,
        v_max=arguments.v_max,
        learn_steps_per_iteration=arguments.learn_steps_per_iteration,
        campaign=Path(arguments.campaign) if arguments.campaign else None,
        checkpoint=Path(arguments.checkpoint) if arguments.checkpoint else None,
        investor=not arguments.no_investor,
        headline_target=arguments.headline_target,
        diagnostics_seed=arguments.diagnostics_seed,
        decimals=arguments.decimals or defaults.decimals,
        resume=arguments.resume,
        output_dir=Path(arguments.out) if arguments.out else defaults.output_dir,
    )


def is_sweep(arguments: argparse.Namespace) -> bool:
    """A sweep is asked for by the plural flags, or implied by ``--report-only``."""
    return bool(arguments.scenarios or arguments.seeds or arguments.report_only)
