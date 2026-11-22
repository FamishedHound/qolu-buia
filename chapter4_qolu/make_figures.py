"""Build every Chapter 4 figure from results that are already on disk.

Every published table *and* every published figure of Chapter 4 is regenerable
by one command from a finished campaign:

===========================================  =======================================
published artefact                           figure family
===========================================  =======================================
Tables 4-7                                   ``table_bars_scenario<k>_<view>``
``scenario_comparison_qolu_buia.pdf``        ``scenario_comparison_<tag>``
``colours_legend_pie.pdf`` + the pies        ``land_preference_<class>``
Figure 1 (the study region, before / after)  ``region_before_after_<tag>``
(no published counterpart)                   ``curves_scenario<k>``,
                                             ``region_converted_<tag>``,
                                             ``land_cover_scenario<k>_seed<n>``,
                                             ``scan_map_<tag>``
===========================================  =======================================

Usage::

    # every figure of one campaign
    python -m chapter4_qolu.make_figures --campaign results/campaign_main

    # including the land-preference pies, which need the land cover
    python -m chapter4_qolu.make_figures --campaign results/campaign_main \\
        --raster lcm2015_gb_1km_dominant_aggregate_class.img --region whole

    # one scenario / seed / view only
    python -m chapter4_qolu.make_figures --campaign results/campaign_main \\
        --scenario 1 --seed 0 --view minmax

    # records without land-cover grids (the published ones): only the
    # figures computed from the metrics
    python -m chapter4_qolu.make_figures --campaign results/campaign_main --no-land-cover

    # the window-scan map (points at the scan folder, not a campaign)
    python -m chapter4_qolu.make_figures --scan results/window_scan_se_20260924

Without ``--out`` the figures land in ``<campaign>/figures`` (or
``<scan>/figures``), next to the numbers that produced them.  Every run writes
``figures_index.md`` listing each figure with its source and the command that
regenerates it.

The land-cover figures (the before / after maps, the region maps and the
land-preference pies) are drawn from the land-cover grids the run records
store.  The records published with this package carry none, because the grids
are derived from the UKCEH Land Cover Map 2015, so for those records the
command stops and says how to regenerate the grids, and ``--no-land-cover``
draws the other figures.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from chapter4_qolu.figures.backend import FigureResult, matplotlib_available
from chapter4_qolu.figures.bars import VIEWS, table_bars
from chapter4_qolu.figures.curves import iteration_curves
from chapter4_qolu.figures.data import (
    LoadedCampaign,
    load_campaign,
    runs_for,
    scenarios_in,
    seeds_in,
)
from chapter4_qolu.figures.index import write_figures_index
from chapter4_qolu.figures.land_preference import land_preference_figures
from chapter4_qolu.figures.maps import land_cover_maps
from chapter4_qolu.figures.region_maps import (
    ORIENTATIONS,
    region_before_after,
    region_converted,
)
from chapter4_qolu.figures.scan_map import scan_score_map
from chapter4_qolu.figures.scenario_comparison import scenario_comparison
from chapter4_qolu.reporting.derived import has_stored_grids, missing_grids_message

__all__ = [
    "campaign_figures",
    "main",
    "region_figures",
    "scan_figures",
]

logger = logging.getLogger(__name__)

#: Design note: the region figures are drawn from the *reference* method's run,
#: because Figure 1 illustrates what the published system did to the landscape.
_REGION_METHOD = "qolu_buia"


def campaign_figures(
    campaign: LoadedCampaign,
    out_dir: Path,
    scenario: int | None = None,
    seed: int | None = None,
    view: str | None = None,
    land_cover: bool = True,
) -> list[FigureResult]:
    """Maps, curves, bar charts and the published scenario comparison.

    ``land_cover=False`` leaves out the before / after maps, the only family
    here that is drawn from the records' land-cover grids.
    """
    scenarios = (
        (int(scenario),) if scenario is not None else scenarios_in(campaign.records)
    )
    views = (view,) if view else VIEWS
    results: list[FigureResult] = [
        scenario_comparison(campaign, out_dir, view=view or "absolute")
    ]
    for item in scenarios:
        cell = [r for r in campaign.records if int(r["scenario"]) == item]
        if land_cover:
            chosen_seeds = (int(seed),) if seed is not None else seeds_in(cell)
            for chosen in chosen_seeds:
                results.append(land_cover_maps(campaign, out_dir, item, chosen))
        results.append(iteration_curves(campaign, out_dir, item))
        for name in views:
            results.append(table_bars(campaign, out_dir, item, name))
    return results


def region_figures(
    campaign: LoadedCampaign,
    out_dir: Path,
    scenario: int = 1,
    seed: int = 0,
    orientation: str = "paper",
    method: str = _REGION_METHOD,
) -> list[FigureResult]:
    """Figure-1 equivalents and the land-preference pies, from one run's grids."""
    from chapter4_qolu.figures.data import grids_by_method

    records = runs_for(campaign.records, scenario=scenario, seed=seed)
    grids = grids_by_method(records)
    if method not in grids:
        if not grids:
            logger.warning(
                "no land-cover grids in %s: the region figures need records "
                "written since run_record stored them",
                campaign.directory,
            )
            return []
        method = next(iter(grids))
    initial, final = grids[method]
    tag = campaign.tag
    command = (
        f"python -m chapter4_qolu.make_figures --campaign {campaign.directory} "
        f"--region --scenario {scenario} --seed {seed}"
    )
    results = [
        region_before_after(
            initial,
            final,
            out_dir,
            tag,
            orientation=orientation,
            source=str(campaign.directory),
            command=command,
        ),
        region_converted(
            initial,
            final,
            out_dir,
            tag,
            orientation=orientation,
            source=str(campaign.directory),
            command=command,
        ),
    ]
    results.extend(
        land_preference_figures(
            {"initial": initial, "final": final},
            out_dir,
            stage="initial",
            source=str(campaign.directory),
            command=command,
        )
    )
    return results


def _require_land_cover(
    campaign: LoadedCampaign, scenario: int | None, seed: int | None
) -> None:
    """Stop, and say how to get the grids back, when the records carry none."""
    lacking = [
        record
        for record in runs_for(campaign.records, scenario=scenario, seed=seed)
        if not has_stored_grids(record)
    ]
    if lacking:
        raise SystemExit(
            "error: "
            + missing_grids_message(
                "Drawing the land-cover figures (the before / after maps, the "
                "study-region maps and the land-preference pies)",
                campaign.directory,
                len(lacking),
            )
            + "  To draw only the figures computed from the metrics (scenario "
            "comparison, curves, bar charts), add --no-land-cover."
        )


def scan_figures(
    scan_dir: Path, out_dir: Path, raster_override: str | None = None
) -> list[FigureResult]:
    """The scan map of a window scan, read from its ``scan_summary.json``.

    The raster is only used for the land / sea backdrop; ``--raster`` overrides
    the path recorded in the summary, which is what a scan run from a temporary
    extraction directory needs.
    """
    summary_path = Path(scan_dir) / "scan_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"no scan_summary.json under {scan_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raster = raster_override or (summary.get("raster") or {}).get("path")
    return [scan_score_map(summary, out_dir, raster_path=raster)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=None, help="campaign results folder")
    parser.add_argument("--scan", default=None, help="window-scan results folder")
    parser.add_argument("--out", default=None, help="where the figures are written")
    parser.add_argument("--scenario", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--view", default=None, choices=VIEWS)
    parser.add_argument(
        "--raster",
        default=None,
        help="the LCM raster the campaign ran on; only needed for figures "
        "drawn from land cover rather than from a run record",
    )
    parser.add_argument(
        "--region",
        nargs="?",
        const="whole",
        default=None,
        metavar="whole|<window>",
        help="also draw the Figure-1 region maps and the land-preference pies",
    )
    parser.add_argument(
        "--map-orientation",
        default="paper",
        choices=ORIENTATIONS,
        help="`paper` transposes the map (west at the top, north on the "
        "left) as the published Figure 1 does; `raster` keeps north up",
    )
    parser.add_argument(
        "--no-land-cover",
        action="store_true",
        help="draw only the figures computed from the metrics (scenario "
        "comparison, curves, bar charts); the land-cover maps, the region "
        "figures and the land-preference pies need the records' land-cover "
        "grids, which the published records do not carry",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not arguments.campaign and not arguments.scan:
        raise SystemExit("give --campaign, --scan, or both")
    if not matplotlib_available():
        logger.warning(
            "matplotlib is not importable: writing figure *data* only (.csv)"
        )

    results: list[FigureResult] = []
    source = Path(arguments.campaign or arguments.scan)
    out_dir = Path(arguments.out) if arguments.out else source / "figures"
    if arguments.campaign:
        campaign = load_campaign(Path(arguments.campaign))
        if not campaign.records:
            raise SystemExit(f"no per-run records under {arguments.campaign}")
        land_cover = not arguments.no_land_cover
        if land_cover:
            _require_land_cover(campaign, arguments.scenario, arguments.seed)
        results.extend(
            campaign_figures(
                campaign,
                out_dir,
                scenario=arguments.scenario,
                seed=arguments.seed,
                view=arguments.view,
                land_cover=land_cover,
            )
        )
        if arguments.region and not land_cover:
            logger.info(
                "--no-land-cover: the region figures and the land-preference "
                "pies are not drawn"
            )
        elif arguments.region:
            results.extend(
                region_figures(
                    campaign,
                    out_dir,
                    scenario=arguments.scenario or 1,
                    seed=arguments.seed or 0,
                    orientation=arguments.map_orientation,
                )
            )
    if arguments.scan:
        results.extend(
            scan_figures(
                Path(arguments.scan), out_dir, raster_override=arguments.raster
            )
        )

    index = write_figures_index(out_dir, results)
    for result in results:
        logger.info("wrote %s (%d file(s))", result.name, len(result.paths))
    logger.info("wrote %s", index)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
