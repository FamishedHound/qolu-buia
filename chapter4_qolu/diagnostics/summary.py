"""Assemble every statistic into the one JSON document of a diagnostics run.

The layout mirrors the question being asked.  ``policy_use_of_the_distribution``
states, as a fact about this code base, that the decision path reads ``Z`` only
through its mean - a statement, not a measurement.  ``pooled`` and ``stakeholders`` then carry the measurement: for
each of the three realised-return readings, the error of ``E[Z]``, the
randomised PIT and its KS test, the empirical coverage of the central 50 % and
90 % intervals, the CRPS with its two skill scores, and the boundary mass.

``policy`` then measures the thing that *reads* the forecast rather than the
forecast itself: per stakeholder, the fraction of appraised candidates approved
and the spread of the quality score across them.  The two are reported together
because neither is sufficient - a stakeholder that approves everything because
it separates nothing and one that approves everything on the merits report the
same approval rate, and only the spread tells them apart.

``verdict`` reduces the pooled headline target to three booleans and one
sentence.  It is written to be readable when the answer is *no*: a distribution
that is badly calibrated, or that has collapsed onto the edges of its support,
is a complete answer and is reported in the same words as a good one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chapter4_qolu.diagnostics.calibration import calibration_statistics
from chapter4_qolu.diagnostics.crps import climatology_crps, crps, crps_skill_score
from chapter4_qolu.diagnostics.errors import (
    boundary_mass,
    error_statistics,
    support_resolution,
    target_clustering,
)
from chapter4_qolu.diagnostics.records import (
    HEADLINE_TARGET,
    NO_SPREAD_RATIO,
    AppraisalLog,
    DiagnosticsLog,
)
from chapter4_qolu.diagnostics.returns import TARGET_DESCRIPTIONS

__all__ = [
    "CALIBRATION_CAVEAT",
    "POLICY_USE_OF_THE_DISTRIBUTION",
    "diagnostics_summary",
    "policy_record",
    "target_statistics",
    "verdict_for",
]

#: How the decision path uses the distribution, stated rather than measured.
#: Every claim here is a claim about code in this package and is pinned by a
#: test in ``tests/test_diagnostics_collect.py``.
POLICY_USE_OF_THE_DISTRIBUTION: dict[str, Any] = {
    "depends_only_on_the_mean": True,
    "quality_score": (
        "Q(s, approve) - Q(s, reject), with Q(s, a) = sum_i z_i p_i(s, a) "
        "= E[Z(s, a)]"
    ),
    "greedy_action": "argmax_a E[Z(s, a)]",
    "consensus": (
        "the signed sum of the stakeholders' quality scores, so the consensus "
        "is a function of the means alone"
    ),
    "risk_sensitivity": (
        "none - no quantile, variance or tail of Z enters a decision anywhere "
        "in this package"
    ),
    "where_the_distribution_acts": (
        "the learning rule only - the loss is the KL divergence against the "
        "projected n-step target distribution, so the shape of Z shapes the "
        "gradients and therefore the mean that is learnt"
    ),
    "consequence": (
        "Z is an internal model, not a decision variable; the statistics below "
        "measure whether that internal model is any good."
    ),
}

#: A KS test this size or larger is called miscalibrated.
CALIBRATION_ALPHA: float = 0.05

#: Design note: the `calibrated` boolean is a threshold on a p-value, and that
#: p-value is not a valid significance test here.  An evaluation rollout logs
#: one appraisal per candidate, 50 candidates share a state within an iteration
#: and consecutive iterations share a landscape, so the sample is strongly
#: dependent and its effective size is far below its nominal one; a KS p-value
#: computed as if it were i.i.d. therefore rejects almost any departure at these
#: sample sizes.  The boolean is kept because a verdict needs one, and this
#: sentence is reported beside it - in both directions, so a *passing* test is
#: not quoted as stronger evidence than it is either.
CALIBRATION_CAVEAT: str = (
    "The KS p-value is reported as a descriptive threshold, not as a "
    "significance test: an evaluation rollout's appraisals are not independent "
    "(50 share a state within an iteration, and consecutive iterations share a "
    "landscape), so the effective sample size is far below the nominal one and "
    "a p-value computed as if the sample were i.i.d. is anti-conservative at "
    "these sizes.  Read the KS statistic, which is an effect size, rather than "
    "the p-value beside it."
)


def target_statistics(
    log: AppraisalLog,
    realised: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Every statistic for one log scored against one realised-return reading."""
    atoms = log.atoms
    probabilities = log.probabilities
    errors = error_statistics(probabilities, atoms, realised)
    calibration = calibration_statistics(probabilities, atoms, realised, rng)

    model = float(crps(probabilities, atoms, realised).mean())
    initialisation = float(crps(log.reference_probabilities, atoms, realised).mean())
    climatology = float(climatology_crps(atoms, realised).mean())
    return {
        "n": errors["n"],
        "mae": errors["mae"],
        "rmse": errors["rmse"],
        "bias": errors["bias"],
        "mean_predicted": errors["mean_predicted"],
        "mean_realised": errors["mean_realised"],
        "sd_realised": errors["sd_realised"],
        "pit": calibration["pit"],
        "coverage": calibration["coverage"],
        "crps": {
            "model": model,
            "reference": {
                "initialisation": initialisation,
                "climatology": climatology,
            },
            "skill": {
                "initialisation": crps_skill_score(model, initialisation),
                "climatology": crps_skill_score(model, climatology),
            },
        },
        "boundary_mass": boundary_mass(probabilities),
        "support": support_resolution(atoms, realised),
        "clustering": target_clustering(log.iteration_index, realised),
    }


def _statistics_for(
    log: AppraisalLog,
    targets: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, Any]:
    record = log.as_record()
    record["targets"] = {
        name: target_statistics(log, values, rng) for name, values in targets.items()
    }
    return record


def policy_record(log: DiagnosticsLog) -> dict[str, Any]:
    """Whether the learnt policy separated the candidates it was shown.

    Two numbers per stakeholder, because neither is sufficient alone:

    * the **approval rate**, the fraction of appraised candidates approved;
    * the **spread of the quality score** ``Q(s, approve) - Q(s, reject)``
      across those candidates.

    An approval rate of exactly 0 or exactly 1 is consistent with a policy that
    has collapsed onto a constant *and* with one that simply agreed with every
    proposal it happened to be shown; the spread tells them apart.  A
    stakeholder is called discriminating only when it does both - varies its
    score and actually splits its verdicts - and the statement below is written
    to read the same way whichever way it comes out.
    """
    entries = []
    for entry in log.logs:
        record = entry.as_record()
        quality = record["quality_score"]
        rate = float(record["approval_rate"])
        entries.append(
            {
                "stakeholder": int(record["stakeholder"]),
                "objective": record["objective"],
                "approval_rate": rate,
                "degenerate_approval": bool(rate in (0.0, 1.0)),
                "quality_score": quality,
                "discriminating": bool(not quality["no_spread"] and 0.0 < rate < 1.0),
            }
        )

    total = len(entries)
    discriminating = sum(1 for entry in entries if entry["discriminating"])
    degenerate = sum(1 for entry in entries if entry["degenerate_approval"])
    no_spread = sum(1 for entry in entries if entry["quality_score"]["no_spread"])
    spreads = [float(entry["quality_score"]["sd"]) for entry in entries]
    statement = (
        f"Of {total} stakeholder(s), {degenerate} approved either every "
        f"candidate or none of them, and {no_spread} gave every candidate the "
        "same quality score to within "
        f"{NO_SPREAD_RATIO:g} of its own magnitude.  The quality-score "
        f"standard deviation runs from {min(spreads):.4g} to "
        f"{max(spreads):.4g} across stakeholders"
        + (
            f", and {discriminating} of {total} both varied that score and "
            "split their verdicts: the policy is state-dependent."
            if discriminating
            else ": no stakeholder both varied that score and split its "
            "verdicts, so the policy separates no candidate from another and "
            "its approval rate is a property of the network rather than of "
            "the states it appraised."
        )
    )
    return {
        "n_stakeholders": total,
        "n_discriminating": discriminating,
        "n_degenerate_approval": degenerate,
        "n_no_spread": no_spread,
        "min_quality_sd": min(spreads),
        "max_quality_sd": max(spreads),
        "state_dependent": bool(discriminating > 0),
        "no_spread_ratio": NO_SPREAD_RATIO,
        "stakeholders": entries,
        "statement": statement,
    }


def verdict_for(headline: dict[str, Any]) -> dict[str, Any]:
    """Four booleans and one paragraph, stated the same way either way round."""
    skill = headline["crps"]["skill"]["climatology"]
    support = headline["support"]
    clustering = headline.get("clustering", {})
    effective = int(clustering.get("effective_n", headline["n"]))
    state_conditional = bool(np.isfinite(skill) and skill > 0.0)
    calibrated = bool(headline["pit"]["ks_p_value"] >= CALIBRATION_ALPHA)
    support_ok = not bool(headline["boundary_mass"]["flagged"])
    resolves = bool(support["resolves_the_signal"])

    parts = [
        (
            f"CRPS skill against climatology is {skill:+.3f}: the return "
            "distribution "
            + (
                "carries state-conditional information, i.e. an internal model "
                "of the return has been learnt."
                if state_conditional
                else "carries no more information than the run's marginal "
                "histogram, i.e. no state-conditional internal model is "
                "evident."
            )
        ),
        (
            "The randomised PIT is consistent with uniformity "
            f"(KS = {headline['pit']['ks_statistic']:.3f}, "
            f"p = {headline['pit']['ks_p_value']:.3g})."
            if calibrated
            else "The randomised PIT departs from uniformity "
            f"(KS = {headline['pit']['ks_statistic']:.3f}, "
            f"p = {headline['pit']['ks_p_value']:.3g}), so the distribution is "
            "not calibrated."
        ),
        (
            "Mass on the two extreme atoms averages "
            f"{headline['boundary_mass']['total_mean']:.4f}"
            + (
                ", so the support holds the distribution."
                if support_ok
                else ", above the "
                f"{headline['boundary_mass']['threshold']:.2f} flag: the "
                "distribution is being clipped by its support and is not well "
                "learnt."
            )
        ),
        (
            f"The atom spacing is delta_z = {support['delta_z']:.3g} against a "
            f"realised return of standard deviation {support['realised_sd']:.3g} "
            f"({support['realised_sd_in_atoms']:.3g} atoms), and the run's "
            f"returns occupy {support['distinct_atoms_used']} of "
            f"{support['n_atoms']} atoms"
            + (
                ", so the support can express the variation being predicted."
                if resolves
                else ": the support is far wider than the signal, so the "
                "categorical head cannot express the variation being "
                "predicted and every statistic above is measuring the "
                "discretisation as much as the model."
            )
        ),
        (
            "The mean absolute error of E[Z], the quantity that actually drives "
            f"decisions, is {headline['mae']:.4f} against a realised return "
            f"with standard deviation {headline['sd_realised']:.4f}."
        ),
        (
            f"{headline['n']} appraisals were scored, but the realised return is "
            "constant within an iteration under this reward reading, so they "
            f"carry {effective} independent observations, not {headline['n']}."
            if clustering.get("constant_within_iteration")
            else (
                f"{headline['n']} appraisals were scored over "
                f"{clustering.get('n_iterations', '?')} iterations and take "
                f"{clustering.get('distinct_realised_values', '?')} distinct "
                "realised values, so the nominal count overstates the "
                "independent information behind every statistic above."
            )
        ),
    ]
    return {
        "n": int(headline["n"]),
        "effective_n": effective,
        "state_conditional": state_conditional,
        "calibrated": calibrated,
        "support_adequate": support_ok,
        "support_resolves_the_signal": resolves,
        "crps_skill_vs_climatology": float(skill),
        "crps_skill_vs_initialisation": float(
            headline["crps"]["skill"]["initialisation"]
        ),
        "calibration_caveat": CALIBRATION_CAVEAT,
        "statement": " ".join(parts),
    }


def diagnostics_summary(
    log: DiagnosticsLog,
    rng: np.random.Generator,
    headline_target: str = HEADLINE_TARGET,
) -> dict[str, Any]:
    """The whole document, ready to be written out as JSON."""
    if headline_target not in TARGET_DESCRIPTIONS:
        raise ValueError(
            f"unknown headline target {headline_target!r}; expected one of "
            f"{sorted(TARGET_DESCRIPTIONS)}"
        )
    per_stakeholder = [
        _statistics_for(entry, entry.realised_returns(), rng) for entry in log.logs
    ]
    pooled = _statistics_for(log.pooled(), log.pooled_returns(), rng)
    return {
        "policy_use_of_the_distribution": dict(POLICY_USE_OF_THE_DISTRIBUTION),
        "rollout": log.as_record(),
        "targets": dict(TARGET_DESCRIPTIONS),
        "headline_target": headline_target,
        "pooled": pooled,
        "stakeholders": per_stakeholder,
        # A statement about the *policy* rather than about the forecast: the
        # CRPS answers "is Z any good", and this answers "did the thing that
        # reads Z ever behave differently on two different candidates".
        "policy": policy_record(log),
        "verdict": verdict_for(pooled["targets"][headline_target]),
    }
