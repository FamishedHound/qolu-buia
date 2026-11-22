"""The markdown document of one diagnostics run.

One document, in the order the question is asked: what the distribution is used
for, what it was scored against, the numbers, and a verdict written so that it
reads the same way whether the answer is good or bad.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

__all__ = ["DEFAULT_DECIMALS", "render_diagnostics_report"]

DEFAULT_DECIMALS: int = 4

_STATISTIC_ROWS: tuple[tuple[str, str], ...] = (
    ("n", "appraisals scored"),
    ("mae", "MAE of E[Z]"),
    ("rmse", "RMSE of E[Z]"),
    ("bias", "bias of E[Z] (predicted - realised)"),
    ("sd_realised", "SD of the realised return"),
)


def _number(value: Any, decimals: int) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.{decimals}f}"


def _headline_table(targets: dict[str, Any], decimals: int) -> list[str]:
    names = list(targets)
    header = "| Statistic | " + " | ".join(f"`{name}`" for name in names) + " |"
    rule = "| --- |" + " --- |" * len(names)
    lines = [header, rule]
    for key, label in _STATISTIC_ROWS:
        cells = [_number(targets[name][key], decimals) for name in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for key, label in (
        ("effective_n", "independent observations (effective n)"),
        ("distinct_realised_values", "distinct realised return values"),
    ):
        lines.append(
            f"| {label} | "
            + " | ".join(
                _number(targets[name].get("clustering", {}).get(key), decimals)
                for name in names
            )
            + " |"
        )
    lines.append(
        "| PIT KS statistic | "
        + " | ".join(
            _number(targets[name]["pit"]["ks_statistic"], decimals) for name in names
        )
        + " |"
    )
    lines.append(
        "| PIT KS p-value | "
        + " | ".join(
            _number(targets[name]["pit"]["ks_p_value"], decimals) for name in names
        )
        + " |"
    )
    for index, entry in enumerate(targets[names[0]]["coverage"]):
        level = round(100 * float(entry["level"]))
        for key, what in (
            ("empirical", "coverage"),
            ("mean_width", "mean interval width"),
        ):
            lines.append(
                f"| {what} of the central {level} % interval"
                + (f" (nominal {entry['nominal']:.2f})" if key == "empirical" else "")
                + " | "
                + " | ".join(
                    _number(targets[name]["coverage"][index][key], decimals)
                    for name in names
                )
                + " |"
            )
    lines.append(
        "| CRPS | "
        + " | ".join(
            _number(targets[name]["crps"]["model"], decimals) for name in names
        )
        + " |"
    )
    for reference, label in (
        ("initialisation", "CRPS skill vs the network at initialisation"),
        ("climatology", "**CRPS skill vs climatology**"),
    ):
        lines.append(
            f"| {label} | "
            + " | ".join(
                _number(targets[name]["crps"]["skill"][reference], decimals)
                for name in names
            )
            + " |"
        )
    lines.append(
        "| boundary-atom mass (atoms 0 and N-1) | "
        + " | ".join(
            _number(targets[name]["boundary_mass"]["total_mean"], decimals)
            for name in names
        )
        + " |"
    )
    lines.append(
        "| realised returns outside the support | "
        + " | ".join(
            _number(targets[name]["pit"]["out_of_support_fraction"], decimals)
            for name in names
        )
        + " |"
    )
    for key, label in (
        ("delta_z", "atom spacing delta_z"),
        ("realised_sd_in_atoms", "SD of the realised return, in atoms"),
        ("distinct_atoms_used", "atoms the realised returns land on"),
        ("resolves_the_signal", "support resolves the signal"),
    ):
        lines.append(
            f"| {label} | "
            + " | ".join(
                _number(targets[name]["support"][key], decimals) for name in names
            )
            + " |"
        )
    return lines


def _stakeholder_table(
    stakeholders: list[dict[str, Any]], headline: str, decimals: int
) -> list[str]:
    lines = [
        (
            "| Stakeholder | Objective | MAE of E[Z] | CRPS | CRPS skill vs "
            "climatology | PIT KS p | boundary mass |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in stakeholders:
        target = entry["targets"][headline]
        lines.append(
            "| {index} | {objective} | {mae} | {crps} | {skill} | {p} | {mass} |".format(
                index=entry["stakeholder"],
                objective=entry["objective"],
                mae=_number(target["mae"], decimals),
                crps=_number(target["crps"]["model"], decimals),
                skill=_number(target["crps"]["skill"]["climatology"], decimals),
                p=_number(target["pit"]["ks_p_value"], decimals),
                mass=_number(target["boundary_mass"]["total_mean"], decimals),
            )
        )
    return lines


def _policy_lines(summary: dict[str, Any], decimals: int) -> list[str]:
    """The state-dependence table: approval rate *and* quality-score spread.

    Section 4 of this document scores the forecast; this scores the policy that
    reads it.  A constant policy and a discriminating one can report the same
    approval rate, so the rate is printed beside the spread of
    ``Q(s, approve) - Q(s, reject)`` rather than on its own.

    A summary without a ``policy`` key gets no section 5, so ``--resume`` can
    re-render any stored summary.
    """
    policy = summary.get("policy")
    if not policy:
        return []
    lines = [
        "## 5. Policy state-dependence",
        "",
        (
            "The approval rate alone cannot distinguish a stakeholder that "
            "approves everything because it separates nothing from one that "
            "approves everything on the merits.  The spread of the quality "
            "score across the candidates appraised can: a policy that puts "
            "every candidate in the same place has a standard deviation of "
            "zero."
        ),
        "",
        (
            "| Stakeholder | Objective | approval rate | mean quality score | "
            "**SD of the quality score** | SD within an iteration | min | max "
            "| SD / mean abs. | discriminates |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in policy["stakeholders"]:
        quality = entry["quality_score"]
        cells = [
            str(entry["stakeholder"]),
            str(entry["objective"]),
            _number(entry["approval_rate"], decimals),
            _number(quality["mean"], decimals),
            _number(quality["sd"], decimals),
            _number(quality["within_iteration_sd"], decimals),
            _number(quality["min"], decimals),
            _number(quality["max"], decimals),
            _number(quality["spread_ratio"], decimals),
            "yes" if entry["discriminating"] else "no",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        policy["statement"],
        "",
        (
            f"A stakeholder counts as *discriminating* when its quality-score "
            f"spread exceeds {policy['no_spread_ratio']:g} of the score's own "
            "mean magnitude **and** its approval rate lies strictly between 0 "
            "and 1.  The threshold is a reporting convenience; the raw spread "
            "is in the table so a reader can apply their own."
        ),
        "",
    ]
    return lines


def _learner_lines(summary: dict[str, Any]) -> list[str]:
    """One sentence on whether the double-Q target was ever synchronised.

    ``target_sync_interval`` is Algorithm 1's ``t_n``, which Table 4.1 never
    gives, and a value longer than the run's gradient budget silently leaves
    ``theta^-`` at its random initialisation for the whole run.  That changes
    what every number above is a diagnosis *of*, so it belongs in the document
    rather than only in the JSON.

    Design note (per stakeholder, never in total): the stakeholders learn
    independently and the gradient budget is a *per agent* number, so a total
    sync count printed against it would divide a three-agent numerator by a
    one-agent denominator.  Both counts are per agent here, and the number of
    agents is named.

    A summary without a ``learner`` block gets no such sentence, so
    ``--resume`` can re-render any stored summary.
    """
    learner = summary.get("learner")
    if not learner:
        return []
    interval = learner["target_sync_interval"]
    syncs = learner.get("target_syncs") or [learner["total_target_syncs"]]
    steps = learner.get("gradient_steps") or []
    budget = f" took {max(steps)} gradient steps each and" if steps else ""
    if learner["total_target_syncs"]:
        verdict = (
            f"synchronised their target networks {max(syncs)} time(s) each, so "
            "the double-Q bootstrap tracked the online network"
        )
    else:
        verdict = (
            "**never** synchronised their target networks, so every bootstrap "
            "in the run was evaluated on randomly initialised weights"
        )
    return [
        (
            f"Target-network schedule: `t_n` = {interval} gradient steps; the "
            f"{len(syncs)} stakeholder(s){budget} {verdict}."
        ),
        "",
    ]


def render_diagnostics_report(
    summary: dict[str, Any], decimals: int = DEFAULT_DECIMALS
) -> str:
    """Render the whole diagnostics summary as a markdown document."""
    policy = summary["policy_use_of_the_distribution"]
    rollout = summary["rollout"]
    headline = summary["headline_target"]
    verdict = summary["verdict"]
    generated = summary.get(
        "generated", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    )

    lines: list[str] = [
        "# QOLU return-distribution diagnostics",
        "",
        f"Generated {generated}.  Tag `{summary.get('tag', '-')}`.",
        "",
        "## 1. How the distribution is used in the final algorithm",
        "",
        (
            "The decision path reads the categorical return distribution "
            "`Z(s, a)` **only through its mean**:"
        ),
        "",
        f"* quality score: {policy['quality_score']};",
        f"* greedy action: {policy['greedy_action']};",
        f"* consensus: {policy['consensus']};",
        f"* risk sensitivity: {policy['risk_sensitivity']}.",
        "",
        (
            "Two distributions with the same `E[Z]` and different shape "
            "therefore produce the same action, the same quality score and the "
            "same consensus decision.  Where the distribution *does* act is "
            f"{policy['where_the_distribution_acts']}.  {policy['consequence']}"
        ),
        "",
        "## 2. What the distribution was scored against",
        "",
        (
            "This environment has no terminal state inside a run, so three "
            "readings of the realised return are reported side by side and "
            "never mixed:"
        ),
        "",
    ]
    for name, description in summary["targets"].items():
        marker = " **(headline)**" if name == headline else ""
        lines.append(f"* `{name}`{marker} - {description}.")
    lines += [
        "",
        (
            f"Rollout: {rollout['iterations']} iterations x "
            f"{rollout['samples_per_iteration']} appraisals per iteration x "
            f"{rollout['n_stakeholders']} stakeholder(s), reward reading "
            f"`{rollout['qolu_reward_mode']}`, epsilon = "
            f"{rollout['epsilon']:.2f}, investor attached: "
            f"{'yes' if rollout['investor'] else 'no'}."
        ),
        "",
    ]
    lines += _learner_lines(summary)
    lines += [
        "## 3. Accuracy and error of the learnt distribution (pooled)",
        "",
    ]
    lines += _headline_table(summary["pooled"]["targets"], decimals)
    lines += [
        "",
        (
            "Read the support rows column by column.  The three targets live on "
            "different scales, and `n_step_target` is the one the network is "
            "actually regressed onto, so a support that cannot resolve *that* "
            "column is a statement about the algorithm as published - 51 atoms "
            "on [-200, 200] against this environment's reward scale - rather "
            "than about this particular run."
        ),
        "",
        f"## 4. Per stakeholder, scored against `{headline}`",
        "",
    ]
    lines += _stakeholder_table(summary["stakeholders"], headline, decimals)
    lines += ["", *_policy_lines(summary, decimals)]
    lines += [
        "## 6. Verdict" if summary.get("policy") else "## 5. Verdict",
        "",
        verdict["statement"],
        "",
        verdict.get("calibration_caveat", ""),
        "",
        "| Question | Answer |",
        "| --- | --- |",
        (
            "| Is `Z` state-conditional (positive **CRPS skill** vs "
            "climatology)? | "
            f"{'yes' if verdict['state_conditional'] else 'no'} |"
        ),
        (
            "| Is `Z` calibrated (randomised PIT consistent with uniform)? | "
            f"{'yes' if verdict['calibrated'] else 'no'} |"
        ),
        (
            "| Is the atom support adequate (low **boundary**-atom mass)? | "
            f"{'yes' if verdict['support_adequate'] else 'no'} |"
        ),
        (
            "| Is the atom spacing fine enough to express the signal? | "
            f"{'yes' if verdict['support_resolves_the_signal'] else 'no'} |"
        ),
        (
            "| Does the policy depend on anything but `E[Z]`? | "
            f"{'yes' if not policy['depends_only_on_the_mean'] else 'no'} |"
        ),
        "",
        (
            "The CRPS skill against climatology is the number that answers the "
            "first question of the table above: climatology always emits the "
            "run's own marginal histogram of realised returns, so it uses no state "
            "information at all, and beating it is exactly the claim that a "
            "state-conditional internal model has been learnt."
        ),
        "",
    ]
    return "\n".join(lines) + "\n"
