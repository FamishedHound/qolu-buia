"""The stage-B and stage-C sections of ``sensitivity.md``.

Stage B confirms the screening's finalists over four scenarios, three seeds and
both consensus readings; stage C re-runs the single winner at five seeds and is
the deliverable.  Both are rendered here, away from the document's assembly, so
that :mod:`chapter4_qolu.reporting.sensitivity_report` stays a table of
contents rather than a wall of formatting.

The 12-cell count is always printed **twice** - tie-aware and alphabetical -
because the two disagree exactly where ties matter, and a reader who is shown
only one of them cannot tell which they are looking at.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chapter4_qolu.reporting.scan_tables import scenario_tables
from chapter4_qolu.reporting.tables import DEFAULT_DECIMALS

__all__ = [
    "closest_section",
    "stage_a_row",
    "stage_a_score_text",
    "stage_b_overview",
    "stage_c_section",
]

#: The four methods a stage-B cell compares (QOLU + BUIA and three baselines).
_METHODS_PER_CELL = 4


def stage_b_overview(
    stage_b: Mapping[str, Any],
    stage_a_rows: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    campaigns = stage_b.get("campaigns") or []
    lines = [
        "## Stage B - confirmation",
        "",
        (
            "The top configurations re-run over **all four scenarios**, seeds "
            + ", ".join(str(seed) for seed in stage_b.get("seeds", []))
            + ", and **both** readings of the consensus rule - plain (the "
            "published unweighted sum of quality scores) and weighted "
            "(`scale_by_scenario_weight=True`, `weighted_consensus=True`, the "
            "only reading in which the scenario weights reach QOLU+BUIA and "
            "P-MADDPG at all).  Each campaign gets the ordinary documents "
            "(`summary.json`, `tables.md`, `tables.tex`, `pattern_check.md`, "
            "`per_seed.csv`)."
        ),
        "",
        (
            "The 12 (scenario x metric) cells are counted **twice**.  "
            "*Tie-aware* works from the values: on an emphasised metric "
            "QOLU+BUIA must be strictly ahead, elsewhere within a tie of the "
            "best, and the baselines must be ordered P-MADDPG >= LToS >= MitM.  "
            "*Alphabetical* is the campaign's own reading, taken from "
            "`summary.json`'s ranking, where ties are broken by method name - "
            "which silently hands every tie to `ltos` and takes it from "
            "`qolu_buia`.  They are reported side by side because they "
            "disagree exactly where ties matter."
        ),
        "",
    ]
    if not campaigns:
        lines.extend(["No stage-B records were found.", ""])
        return lines
    expected = (
        len(stage_b.get("scenarios") or [])
        * len(stage_b.get("seeds") or [])
        * _METHODS_PER_CELL
    )
    lines.extend(
        [
            (
                "| Stage-A rank | Configuration | Stage-A score | Reading | "
                "Runs | Cells (tie-aware) | Cells (alphabetical) | "
                "MAE (48 cells) |"
            ),
            "| --- |" + " --- |" * 7,
        ]
    )
    partial = False
    for entry in campaigns:
        configuration = entry.get("configuration") or {}
        mae = (entry.get("value_mae") or {}).get(
            entry.get("closeness_view", "absolute")
        )
        found = int(entry.get("n_runs", 0))
        short = bool(expected and found < expected)
        partial = partial or short
        lines.append(
            "| {rank} | `{label}` | {stage_a} | {reading} | {runs} | "
            "{tie} of {total} | {alpha} of {total} | {mae} |".format(
                rank=entry.get("rank", "-"),
                label=configuration.get("label", "?"),
                stage_a=stage_a_score_text(
                    stage_a_rows, str(configuration.get("label", ""))
                ),
                reading=entry.get("reading", "?"),
                runs=(
                    (
                        f"**{found} of {expected}**"
                        if short
                        else f"{found} of {expected}"
                    )
                    if expected
                    else str(found)
                ),
                tie=entry.get("matched_cells_tie_aware", 0),
                alpha=entry.get("matched_cells_alphabetical", 0),
                total=entry.get("total_cells", 0),
                mae="-" if mae is None else f"{float(mae):.3f}",
            )
        )
    if partial:
        lines.extend(
            [
                "",
                (
                    "**Some campaigns above are incomplete** (their `Runs` "
                    "cell is in bold).  A campaign missing runs still gets a "
                    "cell count, and that count is *not* comparable with a "
                    "finished one: a cell whose four methods are not all "
                    "present cannot reproduce an ordering, so a partial "
                    "campaign reads as a failure whether or not it is one.  "
                    "Finish the stage, or read only the complete rows."
                ),
            ]
        )
    lines.extend(
        [
            "",
            (
                "`Stage-A rank` is the configuration's place in the screening "
                "order above, and `Stage-A score` its screening numbers "
                "(`First` / `Tau` / `MAE`) - printed here so a finalist that "
                "confirms badly can be told from one that screened badly."
            ),
            "",
        ]
    )
    return lines


def stage_a_row(
    rows: Sequence[Mapping[str, Any]], label: str
) -> Mapping[str, Any] | None:
    """The stage-A entry of one configuration, by label."""
    return next(
        (row for row in rows if str(row["configuration"]["label"]) == label), None
    )


def stage_a_score_text(rows: Sequence[Mapping[str, Any]], label: str) -> str:
    row = stage_a_row(rows, label)
    if row is None:
        return "-"
    score = row["score"]
    mae = row.get("value_mae")
    return (
        f"{int(score['first_place_count'])}/3, "
        f"{float(score['kendall_sum']):+.3f}, "
        + ("-" if mae is None else f"{float(mae):.3f}")
    )


def stage_c_section(
    stage_c: Mapping[str, Any] | None, decimals: int = DEFAULT_DECIMALS
) -> list[str]:
    """The deliverable campaign: the winning configuration at five seeds."""
    if not stage_c:
        return []
    configuration = stage_c.get("configuration") or {}
    mae = (stage_c.get("value_mae") or {}).get(
        stage_c.get("closeness_view", "absolute")
    )
    lines = [
        "## Stage C - the deliverable",
        "",
        (
            f"The single best configuration, `{configuration.get('label', '?')}` "
            f"({configuration.get('description', '')}), re-run at seeds "
            + ", ".join(str(seed) for seed in stage_c.get("seeds", []))
            + f" under the {stage_c.get('reading', '')} consensus reading: "
            f"{stage_c.get('n_runs', 0)} runs, full campaign documents in "
            f"`{stage_c.get('campaign_dir', '')}`."
        ),
        "",
        (
            f"{stage_c.get('matched_cells_tie_aware', 0)} of "
            f"{stage_c.get('total_cells', 0)} (scenario x metric) cells "
            "reproduce the published ordering under the tie-aware reading, "
            f"{stage_c.get('matched_cells_alphabetical', 0)} of "
            f"{stage_c.get('total_cells', 0)} under the alphabetical one"
            + ("" if mae is None else f"; MAE over the 48 published cells {mae:.3f}")
            + "."
        ),
        "",
        (
            "The paper's standard errors are 0.01-0.03 and it never says how "
            "many runs they come from.  The `n` column of each table below is "
            "this campaign's seed count, so a wider SE here is a statement "
            "about how many runs the published figures would have needed."
        ),
        "",
    ]
    for block in stage_c.get("summary", {}).get("configurations", []):
        for scenario in block.get("scenarios", []):
            lines.extend(scenario_tables(scenario, decimals))
    return lines


def closest_section(
    stage_b: Mapping[str, Any], decimals: int = DEFAULT_DECIMALS
) -> list[str]:
    campaigns = stage_b.get("campaigns") or []
    if not campaigns:
        return []
    entry = campaigns[0]
    configuration = entry.get("configuration") or {}
    window = entry.get("window") or {}
    lines = [
        f"### Closest configuration: `{configuration.get('label', '?')}`",
        "",
        (
            f"{configuration.get('description', '')}, on window "
            f"`{window.get('label', '?')}` ({window.get('description', '')}), "
            f"{entry.get('reading', '')} consensus reading."
        ),
        "",
        (
            f"{entry.get('matched_cells_tie_aware', 0)} of "
            f"{entry.get('total_cells', 0)} (scenario x metric) cells reproduce "
            "the published ordering under the tie-aware reading, "
            f"{entry.get('matched_cells_alphabetical', 0)} of "
            f"{entry.get('total_cells', 0)} under the alphabetical one."
        ),
        "",
    ]
    for campaign_block in entry.get("summary", {}).get("configurations", []):
        for scenario in campaign_block.get("scenarios", []):
            lines.extend(scenario_tables(scenario, decimals))
    return lines
