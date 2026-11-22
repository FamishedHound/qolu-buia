"""Comparison of a campaign's ranking pattern with the published tables.

What the paper's Tables 4-7 actually show, read cell by cell:

* in every table QOLU + BUIA ("US") is first or joint first, P-MADDPG is second
  (or joint first), LToS third and MitM last;
* QOLU + BUIA is *strictly* first only on the metric its scenario emphasises -
  all three in Scenario 1, Land Use in Scenario 2, Woodland in Scenario 3,
  Urban in Scenario 4.  Away from the emphasised metric it ties with P-MADDPG
  (e.g. 0.75 / 0.75 for Woodland in Scenario 2).

Design note (what "agreement" means): a cell agrees when the run reproduces that
*ordering*, not the numbers.  On an emphasised metric that means QOLU + BUIA
ranked first; elsewhere it means QOLU + BUIA in the top two, which is the
published tie.  In both cases the baselines that are present must be ordered
P-MADDPG > LToS > MitM.

The comparison is of the ordering only.  The published cells are quoted to two
decimals on a 0.05 lattice, which is coarser than the between-seed spread of
these metrics, so the values themselves are not compared.

Design note (methods without a published row): the ``no_regulator`` ablation
is left out of the ranking before it is compared, so running it cannot move
QOLU + BUIA's rank in a table that never had it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chapter4_qolu.methods import canonical_method
from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS, REFERENCE_METHOD
from chapter4_qolu.reporting.tables import METHOD_LABELS, METRIC_LABELS

__all__ = [
    "EMPHASISED_METRICS",
    "PUBLISHED_RANKING",
    "PatternCell",
    "check_cell",
    "render_pattern_check",
]

#: The metric each published scenario emphasises (Table 3's weightings).
EMPHASISED_METRICS: Mapping[int, tuple[str, ...]] = {
    1: ("land_use", "woodland", "urban"),
    2: ("land_use",),
    3: ("woodland",),
    4: ("urban",),
}

#: The ordering the published tables show, best first.
PUBLISHED_RANKING: tuple[str, ...] = ("qolu_buia", "pmaddpg", "ltos", "mitm")


@dataclass(frozen=True)
class PatternCell:
    """Whether one (scenario, metric) cell reproduces the published ordering."""

    scenario: int
    metric: str
    emphasised: bool
    ranking: tuple[str, ...]
    reference_rank: int | None
    reference_first: bool
    baselines_in_published_order: bool
    agrees: bool
    verdict: str

    def as_record(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "metric": self.metric,
            "emphasised": self.emphasised,
            "ranking": list(self.ranking),
            "reference_rank": self.reference_rank,
            "reference_first": self.reference_first,
            "baselines_in_published_order": self.baselines_in_published_order,
            "agrees": self.agrees,
            "verdict": self.verdict,
        }


def _baseline_order_holds(
    ranking: Sequence[str], reference: str = REFERENCE_METHOD
) -> bool:
    """``True`` when the baselines present keep the published relative order.

    The ranking may name *variants* (``fixed_ltos``, ``mitm_learned``,
    ``pmaddpg_reg_only``), which the published tables have no row for, so every
    name is mapped onto the published method it stands in for before the
    comparison.  Without that step a variant fails the ``in`` test, the expected
    order shrinks to whichever published names happen to be literal, and the
    check passes vacuously.
    """
    order = [canonical_method(method) for method in ranking]
    expected = [
        method
        for method in PUBLISHED_RANKING
        if method != reference and method in order
    ]
    observed = [method for method in order if method in expected]
    return observed == expected


def check_cell(
    scenario: int,
    metric: str,
    ranking: Sequence[str],
    reference: str = REFERENCE_METHOD,
) -> PatternCell:
    """Compare one cell's ranking with the published expectation."""
    order = tuple(
        method for method in ranking if canonical_method(method) in PUBLISHED_RANKING
    )
    emphasised = metric in EMPHASISED_METRICS.get(int(scenario), ())
    canonical = [canonical_method(method) for method in order]
    rank = canonical.index(reference) + 1 if reference in canonical else None
    reference_first = rank == 1
    baselines_ok = _baseline_order_holds(order, reference)

    if rank is None:
        agrees = False
        verdict = f"{METHOD_LABELS.get(reference, reference)} did not run in this cell"
    else:
        position_ok = reference_first if emphasised else rank <= 2
        agrees = position_ok and baselines_ok
        expectation = (
            "the published tables put it first"
            if emphasised
            else "the published tables put it first or joint first"
        )
        label = METHOD_LABELS.get(reference, reference)
        if agrees:
            verdict = "yes"
        elif not position_ok:
            verdict = f"no - {label} ranked {rank} ({expectation})"
        else:
            verdict = "no - the baselines are not in the published order"
    return PatternCell(
        scenario=int(scenario),
        metric=metric,
        emphasised=emphasised,
        ranking=order,
        reference_rank=rank,
        reference_first=reference_first,
        baselines_in_published_order=baselines_ok,
        agrees=agrees,
        verdict=verdict,
    )


def _ranking_text(ranking: Sequence[str]) -> str:
    return " > ".join(METHOD_LABELS.get(method, method) for method in ranking)


def _published_text(ranking: Sequence[str]) -> str:
    order = {canonical_method(method) for method in ranking}
    present = [method for method in PUBLISHED_RANKING if method in order]
    return (
        " > ".join(METHOD_LABELS.get(method, method) for method in present)
        or "(none of the published methods ran)"
    )


def _configuration_section(
    configuration: Mapping[str, Any], reference: str
) -> tuple[list[str], int, int]:
    lines = [
        f"## Configuration `{configuration['label']}`"
        + (
            f" ({configuration['description']})"
            if configuration.get("description")
            else ""
        ),
        "",
        "| Cell | Metric | Emphasised | Published order | Rebuild order | Agrees? |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    agreeing = 0
    total = 0
    for cell in configuration["scenarios"]:
        scenario = int(cell["scenario"])
        for metric in PUBLISHED_METRICS:
            ranking = cell.get("ranking", {}).get(metric)
            if not ranking:
                continue
            total += 1
            checked = check_cell(scenario, metric, ranking, reference)
            agreeing += int(checked.agrees)
            lines.append(
                "| Scenario {scenario} | {metric} | {emphasis} | {published} | "
                "{rebuild} | {verdict} |".format(
                    scenario=scenario,
                    metric=METRIC_LABELS.get(metric, metric),
                    emphasis="yes" if checked.emphasised else "no",
                    published=_published_text(checked.ranking),
                    rebuild=_ranking_text(checked.ranking),
                    verdict=checked.verdict,
                )
            )
    lines.extend(
        ["", f"{agreeing} of {total} cells agree with the published order.", ""]
    )
    return lines, agreeing, total


def _region_text(campaign: Mapping[str, Any]) -> str:
    """The region sentence, which depends on what the campaign actually ran on.

    Design note: a campaign that pinned a window reports that window; one that
    did not reports the crop the loader chose.  The paper states no extent, so
    the sentence describes the extent that was used and nothing more.
    """
    raster = campaign.get("raster")
    window = campaign.get("window_label")
    if window:
        return (
            f"The study region is `{raster}` cropped to window {window}, which "
            "the window scan selected; the paper states no extent of its own."
        )
    if raster:
        return (
            f"The study region is the extent the loader selected out of "
            f"`{raster}`; the paper states no extent of its own."
        )
    return (
        "The study region is whatever raster this campaign was handed; the "
        "paper states no extent of its own."
    )


def render_pattern_check(summary: Mapping[str, Any]) -> str:
    """The full ``pattern_check.md`` document."""
    reference = str(summary.get("reference_method", REFERENCE_METHOD))
    campaign = summary.get("campaign", {})
    tag = campaign.get("tag", "campaign")
    lines = [
        f"# Campaign `{tag}` - ranking-pattern check",
        "",
        (
            "The published tables rank the methods "
            "**QOLU+BUIA > P-MADDPG > LToS > MitM** in every scenario, with "
            "QOLU+BUIA *strictly* first on the metric its scenario emphasises "
            "(Scenario 1: all three; Scenario 2: Land Use; Scenario 3: "
            "Woodland; Scenario 4: Urban) and tied with P-MADDPG elsewhere.  A "
            "cell below agrees when the rebuild reproduces that ordering: "
            "QOLU+BUIA first on an emphasised metric, in the top two "
            "otherwise, with the baselines present ordered "
            "P-MADDPG > LToS > MitM."
        ),
        "",
        (
            "Rankings are taken from the RAW view; min-max rescaling is affine "
            "per cell and cannot change an ordering."
        ),
        "",
        (
            "The comparison is of the *ordering* only.  The published cells "
            "are quoted to two decimals on a 0.05 lattice and every published "
            "standard error is 0.01, 0.02 or 0.03, so the values themselves "
            f"are not compared.  {_region_text(campaign)}  The three baselines "
            "are re-implementations; see the README for what each one does."
        ),
        "",
    ]
    agreeing = 0
    total = 0
    for configuration in summary["configurations"]:
        section, cells_ok, cells = _configuration_section(configuration, reference)
        lines.extend(section)
        agreeing += cells_ok
        total += cells
    lines.extend(
        [
            "## Overall",
            "",
            (
                f"{agreeing} of {total} cells across every configuration "
                "reproduce the published ranking pattern."
            ),
            "",
        ]
    )
    return "\n".join(lines)
