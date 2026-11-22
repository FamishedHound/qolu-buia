"""``sensitivity.md``: the whole sweep, closest configuration and all.

The document reports the *distribution*, not only the winner: every
configuration gets a row, the score distribution gets its own table, and the
closing section states the selection rule and the counts it was applied to -
the same shape as ``scan.md``, for the same reason.

Rendered from the ``sensitivity_summary.json`` payload
:mod:`chapter4_qolu.run_sensitivity` builds, so the document can be rebuilt
without re-running anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chapter4_qolu.reporting.sensitivity_b2 import composition_note, stage_b2_section
from chapter4_qolu.reporting.sensitivity_stage_a import (
    distribution_table,
    restraint_note,
    stage_a_header,
    stage_a_table,
    top_table,
)
from chapter4_qolu.reporting.sensitivity_stage_b import (
    closest_section,
    stage_a_score_text,
    stage_b_overview,
    stage_c_section,
)
from chapter4_qolu.reporting.tables import DEFAULT_DECIMALS

__all__ = ["render_sensitivity_report"]

#: Section renderers live beside the stage they describe:
#: :mod:`~chapter4_qolu.reporting.sensitivity_stage_a` for the screening,
#: :mod:`~chapter4_qolu.reporting.sensitivity_stage_b` for the confirmation and
#: the deliverable, :mod:`~chapter4_qolu.reporting.sensitivity_b2` for the
#: window-transfer stage.  This module only assembles them.


def _intro(summary: Mapping[str, Any]) -> list[str]:
    settings = summary.get("settings", {})
    return [
        f"# Sensitivity sweep `{summary.get('tag', '')}`",
        "",
        (
            "The package has a handful of configurable readings the published "
            "text does not settle, and each of them changes the numbers.  This "
            "document sweeps them, reports **every** configuration, and names "
            "the one that comes closest to the published Tables 4-7.  The "
            'selection rule is stated under "Method - how the configuration '
            'was chosen" at the end.'
        ),
        "",
        f"* **generated**: {summary.get('generated', '')}",
        f"* **raster**: `{settings.get('raster', '')}`",
        f"* **extent**: {settings.get('grid_shape_text', '')}",
        f"* **iterations**: {settings.get('iterations', '')}",
        f"* **workers**: {settings.get('parallel', '')}, one torch thread each",
        "",
        f"**Region boundary.** {settings.get('region_mask_note', '')}",
        "",
        f"**Land-cover taxonomy.** {settings.get('taxonomy_note', '')}",
        "",
    ]


def _published_section(published: Mapping[str, Any]) -> list[str]:
    if not published:
        return []
    return [
        "## Published normalisation: what are the published numbers *on*?",
        "",
        (
            'The paper says only that its scores are "normalized between 0 and '
            '1" and never says from what.  The campaign tables carry three '
            "readings side by side, and the obvious candidate for the "
            "published one is view (c), the cross-method min-max: within a "
            "column the worst method scores 0 and the best 1.  If that were "
            "the published normalisation, **every** one of the 12 published "
            "(scenario x metric) columns would contain an exact 0 and an exact "
            "1."
        ),
        "",
        (
            f"It does not: {published.get('n_columns_with_zero_and_one', 0)} of "
            f"{published.get('n_columns', 0)} columns do, and the 48 published "
            f"cells run from {published.get('minimum')} to "
            f"{published.get('maximum')} without ever reaching either end of "
            "[0, 1].  **The published tables are therefore not a cross-method "
            "min-max rescaling.**"
        ),
        "",
        (
            "The value-closeness statistic below is consequently computed on "
            f"view (**{published.get('closeness_view', 'absolute')}**), the "
            "absolute-normalised reading with `max_distance = 10 km`, which is "
            "the only one of the three that is bounded in [0, 1], is not "
            "forced to put a 0 and a 1 in every column, and on this land cover "
            "actually lands in the published 0.6-0.9 band.  The min-max "
            "figure is printed next to it, never instead of it."
        ),
        "",
    ]


def _reference_section(references: Sequence[Mapping[str, Any]]) -> list[str]:
    """Variants run once for reference rather than swept."""
    if not references:
        return []
    lines = [
        "### Reference rows (not swept)",
        "",
        (
            "These are run once each and are **not** part of the comparison. "
            "They exist so the report can show what an alternative reading "
            "produces rather than only describing it."
        ),
        "",
        (
            "`pmaddpg_v1` here is the *earlier hyper-parameter set*, not "
            "earlier code: the seeding and noise handling are shared by all "
            "three P-MADDPG variants, so a `v1` run does not bit-reproduce "
            "records written before those were unified and the two should not "
            "be compared cell by cell."
        ),
        "",
        "| Method | Environment | Cap | Reward | Conversions | Why |",
        "| --- |" + " --- |" * 5,
    ]
    for entry in references:
        block = (entry.get("methods") or {}).get(str(entry.get("method")), {})
        lines.append(
            "| `{method}` | `{env}` | {cap} | `{mode}` | {conversions} | {why} |".format(
                method=entry.get("method"),
                env=entry.get("environment"),
                cap=entry.get("cap"),
                mode=entry.get("reward_mode"),
                conversions=(
                    block.get("total_conversions", "-")
                    if entry.get("found")
                    else "not run"
                ),
                why=entry.get("why", ""),
            )
        )
    lines.append("")
    return lines


def _stage_b_done(summary: Mapping[str, Any]) -> bool:
    return bool((summary.get("stage_b") or {}).get("campaigns"))


#: The whole-region environment's label; see ``window_scan.region_window``.
_WHOLE_REGION = "SE_region"


def _whole_region_note(
    configurations: Sequence[Mapping[str, Any]], winner_label: str
) -> list[str]:
    """How the best whole-region configuration scored, and whether it won.

    The extent is the sweep's most consequential axis - the text says 50 x 50
    and Figure 1 shows the whole region - so the reader should not have to
    search a 216-row table to find out how the whole-region reading did.
    """
    region_rows = [
        row
        for row in configurations
        if str(row["configuration"].get("environment")) == _WHOLE_REGION
    ]
    if not region_rows:
        return []
    best = region_rows[0]
    score = best["score"]
    mae = best.get("value_mae")
    won = str(best["configuration"]["label"]) == winner_label
    closest = min(
        (row for row in region_rows if row.get("value_mae") is not None),
        key=lambda row: float(row["value_mae"]),
        default=None,
    )
    lines = [
        "**The whole-region rows.**  The best `"
        + _WHOLE_REGION
        + "` configuration is `"
        + str(best["configuration"]["label"])
        + f"` at {int(score['first_place_count'])}/3, tau "
        + f"{float(score['kendall_sum']):+.3f}, MAE "
        + ("-" if mae is None else f"{float(mae):.3f}")
        + (
            ", and it is the configuration selected above."
            if won
            else ", i.e. it did **not** win: a 50 x 50 window reproduces the "
            "published ordering better than the whole region does."
        ),
    ]
    if closest is not None and str(closest["configuration"]["label"]) != str(
        best["configuration"]["label"]
    ):
        lines[0] += (
            "  The whole region is nevertheless where the *values* come "
            "closest: `"
            + str(closest["configuration"]["label"])
            + f"` reaches MAE {float(closest['value_mae']):.3f} against the "
            "published cells while scoring only "
            + f"{int(closest['score']['first_place_count'])}/3 on the "
            "ordering.  Ordering and closeness therefore select different "
            "configurations, and a report has to say which of the two it "
            "ranks by."
        )
    lines.append("")
    return lines


def _method_section(summary: Mapping[str, Any]) -> list[str]:
    stage_a = summary.get("stage_a", {})
    selection = summary.get("selection", {})
    configurations = stage_a.get("configurations") or []
    best_primary = max(
        (int(entry["score"]["first_place_count"]) for entry in configurations),
        default=0,
    )
    winner = selection.get("winner") or {}
    label = str(winner.get("label", "(none)"))
    stage = str(
        selection.get("winner_stage") or ("B" if _stage_b_done(summary) else "A")
    )
    n_configurations = stage_a.get("n_configurations", len(configurations))
    n_windows = len(stage_a.get("windows") or [])
    chosen_by = (
        "confirmed over four scenarios, three seeds and both consensus "
        "readings in stage B"
        if stage == "B"
        else "**on the stage-A screening alone** - stage B has not been run, "
        "so nothing here is confirmed over more than one scenario and one seed"
    )
    return [
        "## Method - how the configuration was chosen",
        "",
        (
            "Configurations are compared with the published ordering; every "
            f"configuration is listed above.  `{label}` is the best of "
            f"{n_configurations} configurations scored on {n_windows} "
            f"environments ({stage_a.get('n_cells', 0)} cells in all), and it "
            f"is {chosen_by}.  The paper fixes none of the swept quantities, "
            "so the configuration is a property of this sweep."
        ),
        "",
        (
            "Its stage-A screening row (`First` / `Tau` / `MAE`): "
            f"**{stage_a_score_text(configurations, label)}**."
        ),
        "",
        *_whole_region_note(configurations, label),
        (
            "The best primary score reached anywhere in the sweep was "
            f"{best_primary} of 3 - see the distribution table above for how "
            "many configurations reached each score.  The windows themselves "
            "are not searched here: they are the ones the window scan "
            "selected, so the environment axis inherits that scan's choice."
        ),
        "",
        (
            f"With {n_configurations} configurations scored, the leading row "
            "is the maximum of the distribution above and should be read "
            "together with it.  The comparison is of the *ordering*: the "
            "published cells are quoted to two decimals on a 0.05 lattice and "
            "every published standard error is 0.01, 0.02 or 0.03, so the "
            "value-closeness statistic ranks configurations against each other "
            "and nothing more."
        ),
        "",
    ]


def render_sensitivity_report(
    summary: Mapping[str, Any], decimals: int = DEFAULT_DECIMALS
) -> str:
    """The full ``sensitivity.md`` document."""
    stage_a = summary.get("stage_a", {})
    configurations: Sequence[Mapping[str, Any]] = stage_a.get("configurations") or []
    lines: list[str] = []
    lines.extend(_intro(summary))
    lines.extend(_published_section(summary.get("published", {})))
    lines.extend(stage_a_header(stage_a))
    lines.extend(top_table(configurations))
    lines.extend(
        distribution_table(stage_a.get("score_distribution", {}), configurations)
    )
    lines.extend(restraint_note(configurations))
    lines.extend(_reference_section(summary.get("reference") or []))
    lines.extend(["### Every configuration", ""])
    lines.extend(stage_a_table(configurations))
    lines.extend(stage_b_overview(summary.get("stage_b", {}), configurations))
    lines.extend(closest_section(summary.get("stage_b", {}), decimals))
    stage_b2 = summary.get("stage_b2") or {}
    if stage_b2:
        lines.extend(stage_b2_section(stage_b2))
        lines.extend(
            composition_note(
                stage_b2.get("windows") or [],
                str(stage_b2.get("fitted_window", "")),
            )
        )
    lines.extend(stage_c_section(summary.get("stage_c"), decimals))
    lines.extend(_method_section(summary))
    return "\n".join(lines)
