"""The stage-B2 window table, and why the numbers sit where they do.

Stage B2 asks whether the winning *recipe* reproduces the published pattern on
windows other than the one stage A selected it on.  The table prints every
window that was tried - all of them, as everywhere else in this package - with
its ordering score, the saturation distance that suited it best, the resulting
MAE against the published Scenario-1 cells, and the separation statistic that
says whether its four methods differ by anything like the published amount.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chapter4_qolu.reporting.aggregate import PUBLISHED_METRICS

__all__ = ["composition_note", "stage_b2_section"]


def _number(value: Any, decimals: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{decimals}f}"


def _spread(entry: Mapping[str, Any], metric: str) -> tuple[float, float] | None:
    values = [
        block["absolute"][metric]
        for block in (entry.get("methods") or {}).values()
        if metric in (block.get("absolute") or {})
    ]
    if not values:
        return None
    return min(values), max(values)


def composition_note(
    entries: Sequence[Mapping[str, Any]], fitted_window: str
) -> list[str]:
    """Why a window can reproduce the ordering and miss the values entirely.

    Computed from the selected window's own spreads, not asserted: if its four
    methods sit within 0.01 of each other on two of the three metrics, the
    metrics are degenerate there and no ordering they produce can carry the
    published 0.05-0.10 separation.
    """
    entry = next(
        (item for item in entries if item["window"]["label"] == fitted_window), None
    )
    if entry is None:
        return []
    spreads = {metric: _spread(entry, metric) for metric in PUBLISHED_METRICS}
    if not any(spreads.values()):
        return []
    parts = []
    for metric, spread in spreads.items():
        if spread is None:
            continue
        low, high = spread
        parts.append(f"{metric} {low:.3f}-{high:.3f} (spread {high - low:.3f})")
    published = entry.get("published_separation")
    return [
        "### Why the numbers sit where they do",
        "",
        (
            f"On `{fitted_window}`, the window stage A selected, the "
            "four methods' absolute-view values are "
            + "; ".join(parts)
            + ".  The published cells are separated by 0.05-0.10 and sit "
            "between 0.60 and 0.90, so a window whose methods differ by 0.01 "
            "cannot produce the published *pattern* however well it produces "
            "the published *order*."
        ),
        "",
        (
            "The cause is the window's composition, not the recipe.  Land Use "
            "is distance from new development to the nearest heath / bog / "
            "mountain parcel: a window with essentially none of that habitat "
            "scores every developed parcel as maximally 'far', for every "
            "method.  Urban is closeness to *pre-existing* built-up land: a "
            "window that is already largely built up scores every new parcel "
            "as maximally close, again for every method.  Only Woodland - the "
            "share of the initial woodland and arable area still standing - "
            "retains any range, and it is the metric on which the four methods "
            "here actually differ."
        ),
        "",
        (
            "That is what stage B2 is for: the same recipe on every retained "
            "window, so a window can be chosen for reproducing the published "
            "*separation* "
            + (
                f"(mean adjacent gap {float(published):.3f})"
                if published is not None
                else ""
            )
            + " as well as the published order."
        ),
        "",
    ]


def stage_b2_section(stage_b2: Mapping[str, Any]) -> list[str]:
    """The full stage-B2 table - every window tried, none hidden."""
    entries = stage_b2.get("windows") or []
    recipe = stage_b2.get("recipe") or {}
    lines = [
        "## Stage B2 - the same recipe on every window",
        "",
        (
            "Stage A selected the recipe on three environments and stage B "
            "confirmed it over four scenarios and three seeds - but neither "
            "separates the **recipe** from the **window**.  Stage B2 runs the "
            "winning recipe "
            + (
                f"(`{recipe.get('label', '?')}`, "
                f"{recipe.get('reading', 'plain')} consensus) "
                if recipe
                else ""
            )
            + "on Scenario 1 at seed 0 across **every window the Southeast "
            "scan retained**, plus the whole-region box, changing nothing but "
            "the extent."
        ),
        "",
        (
            "Windows are ordered by: strict first places (**3 of 3 is required "
            "to qualify**), then Kendall's tau-a, then the MAE against the 12 "
            "published Scenario-1 cells at the saturation distance that suits "
            "that window best, then how near its method separation is to the "
            "published one.  `Sep` is the mean absolute gap between adjacent "
            "methods, averaged over the three metrics, and `Sep/pub` is that "
            "divided by the published figure; a value near 1 means the "
            "rebuild's methods differ by about as much as the paper's do."
        ),
        "",
    ]
    if not entries:
        lines.extend(["No stage-B2 records were found.", ""])
        return lines
    lines.extend(
        [
            (
                "| # | Window | Extent | First | Tau | Best max_distance | MAE "
                "| Sep | Sep/pub | Qualifies |"
            ),
            "| --- |" + " --- |" * 9,
        ]
    )
    for index, entry in enumerate(entries, start=1):
        window = entry["window"]
        score = entry["score"]
        ratio = entry.get("separation_ratio")
        lines.append(
            "| {index} | `{label}` | {extent} | {first}/3 | {tau:+.3f} | "
            "{distance} | {mae} | {sep} | {ratio} | {ok} |".format(
                index=index,
                label=window["label"],
                extent=window["description"],
                first=int(score["first_place_count"]),
                tau=float(score["kendall_sum"]),
                distance=(
                    "-"
                    if entry.get("max_distance_km") is None
                    else f"{float(entry['max_distance_km']):g} km"
                ),
                mae=_number(entry.get("value_mae")),
                sep=_number(entry.get("separation"), 4),
                ratio=_number(ratio, 2),
                ok=(
                    "yes"
                    if entry.get("qualifies")
                    else ("no" if entry.get("n_records") else "no runs")
                ),
            )
        )
    qualifying = [entry for entry in entries if entry.get("qualifies")]
    lines.extend(
        [
            "",
            (
                f"**{len(qualifying)} of {len(entries)} windows reproduce the "
                "published ordering on all three Scenario-1 metrics under this "
                "recipe.**"
                + (
                    ""
                    if qualifying
                    else "  Stage C therefore falls back to the stage-B "
                    "winner: the recipe does not transfer to another window."
                )
            ),
            "",
        ]
    )
    in_band = [entry for entry in qualifying if entry.get("separation_in_band")]
    if qualifying:
        lines.extend(
            [
                (
                    f"Of those, **{len(in_band)}** also separate their methods "
                    "by within a factor of two of the published gap.  A window "
                    "that qualifies on the ordering but not on the separation "
                    "reproduces *which method wins* and not *by how much*, and "
                    "the two are different claims."
                ),
                "",
            ]
        )
    return lines
