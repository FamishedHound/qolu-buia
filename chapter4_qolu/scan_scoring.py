"""How a candidate window is scored against the published Scenario-1 ordering.

Stage 1 of the window scan runs all four methods on Scenario 1 at seed 0 and
asks a single question: how close does this landscape get to the ordering the
paper's Table 4 shows, **QOLU+BUIA > P-MADDPG > LToS > MitM** on Land Use,
Woodland and Urban?

Three numbers answer it, in lexicographic priority:

primary
    how many of the three metrics put QOLU + BUIA first (0-3).  This is the
    claim the paper makes and the one the scan is looking for.

secondary
    the sum over the three metrics of Kendall's tau between the window's
    metric *values* and the published order.  Tau is 1 for the published order, -1 for
    its reverse and 0 for an order that agrees on half the pairs, so this
    separates windows that merely get the winner right from windows that
    reproduce the whole ordering.

tertiary
    QOLU + BUIA's mean cross-method min-max score over the three metrics - the
    same view (c) the campaign tables use.  A tie-break with a *margin* in it:
    of two windows that order the methods identically, the one where
    QOLU + BUIA wins by more comes first.

Design note (unweighted Kendall tau): every pair of methods counts the same,
including pairs among the baselines.  The alternative - weighting the pairs
QOLU + BUIA takes part in more heavily - would make the secondary criterion a
second copy of the primary one.

Design note (ties): the score is computed from the metric *values*, never from a
ranking, because ties are the normal case at the published scale rather than an
edge case.  The 50 x 50 grid with up to ten conversions per iteration saturates
long before iteration 1000, and once it has, every method has produced the same
fully-developed landscape and the three metrics agree to the last bit.  Tied
pairs count neither for nor against the published order (Kendall's tau-a),
a strict first place is what the published tables claim and what
``first_place_count`` counts, and joint first places are reported separately.
The alternative - ranking the methods and breaking ties by name, as
``rank_methods`` does for reproducible *reporting* - would silently award every
tie to the alphabetically first method, which is ``ltos``, and count it against
``qolu_buia``, which sorts last.

Nothing here is a claim that the published *values* can be reproduced; see
:mod:`chapter4_qolu.reporting.pattern`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chapter4_qolu.reporting.aggregate import (
    PUBLISHED_METRICS,
    REFERENCE_METHOD,
    min_max_normalise,
    rank_methods,
)
from chapter4_qolu.reporting.pattern import PUBLISHED_RANKING, check_cell

__all__ = [
    "PUBLISHED_SCENARIO1_ORDER",
    "MetricRanking",
    "WindowScore",
    "count_pattern_matches",
    "kendall_tau",
    "kendall_tau_on_values",
    "score_window",
]

#: The ordering Table 4 shows, best first.  Identical to
#: :data:`chapter4_qolu.reporting.pattern.PUBLISHED_RANKING`; re-exported under
#: a scenario-specific name because stage 1 only ever runs Scenario 1, where
#: the literal and the weighted readings of the scenario weights coincide.
PUBLISHED_SCENARIO1_ORDER: tuple[str, ...] = PUBLISHED_RANKING

#: Design note: metric values within this of each other count as identical.  The
#: five metrics are means over a 2,500-parcel grid, so a genuine difference
#: between two methods is of order 1e-3; 1e-12 only catches cells that are
#: literally the same number (usually an exact 0.0 after saturation).
_FLAT_TOLERANCE: float = 1e-12


def kendall_tau(ranking: Sequence[str], reference: Sequence[str]) -> float:
    """Kendall's tau-a between two strict orderings of the same items."""
    observed = list(ranking)
    expected = list(reference)
    if sorted(observed) != sorted(expected):
        raise ValueError(f"rankings are over different items: {observed} vs {expected}")
    n = len(observed)
    if n < 2:
        return 0.0
    position = {name: index for index, name in enumerate(observed)}
    concordant = discordant = 0
    for first in range(n):
        for second in range(first + 1, n):
            left, right = expected[first], expected[second]
            if position[left] < position[right]:
                concordant += 1
            else:
                discordant += 1
    pairs = n * (n - 1) // 2
    return (concordant - discordant) / pairs


def kendall_tau_on_values(
    values: Mapping[str, float],
    published: Sequence[str] = PUBLISHED_SCENARIO1_ORDER,
) -> float:
    """Kendall's tau-a between one metric's values and the published order.

    Every pair of methods is either concordant (the higher-scoring method is
    the one the published tables put first), discordant, or **tied** - and a
    tied pair contributes nothing to either sum while still counting in the
    denominator.  A metric on which every method scored the same therefore
    scores exactly 0: no agreement and no disagreement, which is the neutral
    reading of a landscape that every method developed identically.
    """
    names = [name for name in published if name in values]
    n = len(names)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    for first in range(n):
        for second in range(first + 1, n):
            # ``published`` is ordered best first, so the published claim is
            # values[names[first]] > values[names[second]].
            left = float(values[names[first]])
            right = float(values[names[second]])
            if left > right:
                concordant += 1
            elif left < right:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) // 2)


@dataclass(frozen=True)
class MetricRanking:
    """How one metric of one window compares with the published ordering."""

    metric: str
    ranking: tuple[str, ...]
    #: The reference scored strictly higher than every other method - the
    #: published claim.
    reference_first: bool
    tau: float
    reference_minmax: float
    #: Every method scored the same on this metric, so the "ranking" above is
    #: only the tie-break by method name and carries no information.
    flat: bool = False
    #: The reference is among the joint-best (but perhaps not alone there).
    reference_joint_first: bool = False
    #: How many of the method pairs scored identically.
    tied_pairs: int = 0
    #: ``ranking`` with methods that scored the same collected into one group,
    #: best group first - the only faithful way to print the order.
    groups: tuple[tuple[str, ...], ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "ranking": list(self.ranking),
            "groups": [list(group) for group in self.groups],
            "reference_first": self.reference_first,
            "reference_joint_first": self.reference_joint_first,
            "kendall_tau": self.tau,
            "reference_minmax": self.reference_minmax,
            "flat": self.flat,
            "tied_pairs": self.tied_pairs,
        }


@dataclass(frozen=True)
class WindowScore:
    """The three stage-1 criteria for one candidate window."""

    first_place_count: int
    kendall_sum: float
    reference_minmax_mean: float
    metrics: tuple[MetricRanking, ...]

    @property
    def flat_metrics(self) -> int:
        """How many of the metrics separated no methods at all."""
        return sum(int(item.flat) for item in self.metrics)

    @property
    def joint_first_count(self) -> int:
        """Metrics on which the reference was among the joint-best."""
        return sum(int(item.reference_joint_first) for item in self.metrics)

    @property
    def sort_key(self) -> tuple[int, float, float]:
        """Lexicographic key: best window last under ``sorted``, first under reverse."""
        return (
            self.first_place_count,
            self.kendall_sum,
            self.reference_minmax_mean,
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "first_place_count": self.first_place_count,
            "joint_first_count": self.joint_first_count,
            "kendall_sum": self.kendall_sum,
            "reference_minmax_mean": self.reference_minmax_mean,
            "flat_metrics": self.flat_metrics,
            "metrics": [metric.as_record() for metric in self.metrics],
        }


def _tie_groups(
    values: Mapping[str, float], ranking: Sequence[str]
) -> tuple[tuple[str, ...], ...]:
    """``ranking`` with equal-scoring methods collected into one group."""
    groups: list[list[str]] = []
    for name in ranking:
        value = float(values[name])
        if groups and abs(float(values[groups[-1][0]]) - value) <= _FLAT_TOLERANCE:
            groups[-1].append(name)
        else:
            groups.append([name])
    return tuple(tuple(sorted(group)) for group in groups)


def _tied_pairs(
    values: Mapping[str, float], published: Sequence[str] = PUBLISHED_SCENARIO1_ORDER
) -> int:
    """How many pairs of the published methods scored identically."""
    names = [name for name in published if name in values]
    return sum(
        1
        for first in range(len(names))
        for second in range(first + 1, len(names))
        if abs(float(values[names[first]]) - float(values[names[second]]))
        <= _FLAT_TOLERANCE
    )


def score_window(
    values: Mapping[str, Mapping[str, float]],
    reference: str = REFERENCE_METHOD,
    published: Sequence[str] = PUBLISHED_SCENARIO1_ORDER,
) -> WindowScore:
    """Score one window from ``values[metric][method]`` (RAW metric values)."""
    rankings: list[MetricRanking] = []
    for metric in PUBLISHED_METRICS:
        per_method = {
            name: float(value) for name, value in (values.get(metric) or {}).items()
        }
        if not per_method:
            continue
        # ``rank_methods`` breaks ties by name; the ranking is kept for the
        # report only.  Every *number* below comes from the values.
        ranking = rank_methods(per_method)
        scaled = min_max_normalise(per_method)
        highest = max(per_method.values())
        lowest = min(per_method.values())
        flat = (highest - lowest) <= _FLAT_TOLERANCE
        at_the_top = sum(
            1 for value in per_method.values() if highest - value <= _FLAT_TOLERANCE
        )
        reference_value = per_method.get(reference)
        joint_first = (
            not flat
            and reference_value is not None
            and highest - reference_value <= _FLAT_TOLERANCE
        )
        rankings.append(
            MetricRanking(
                metric=metric,
                ranking=ranking,
                # The published tables claim QOLU + BUIA is *strictly* first on
                # an emphasised metric, so sharing the top spot is not a win.
                reference_first=joint_first and at_the_top == 1,
                reference_joint_first=joint_first,
                tau=kendall_tau_on_values(per_method, published),
                reference_minmax=float(scaled.get(reference, 0.0)),
                flat=flat,
                tied_pairs=_tied_pairs(per_method, published),
                groups=_tie_groups(per_method, ranking),
            )
        )

    if not rankings:
        return WindowScore(0, 0.0, 0.0, ())
    return WindowScore(
        first_place_count=sum(int(item.reference_first) for item in rankings),
        kendall_sum=float(sum(item.tau for item in rankings)),
        reference_minmax_mean=float(
            sum(item.reference_minmax for item in rankings) / len(rankings)
        ),
        metrics=tuple(rankings),
    )


def count_pattern_matches(
    summary: Mapping[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    """How many (scenario, metric) cells of a campaign match the published order.

    The same rule :mod:`chapter4_qolu.reporting.pattern` applies in
    ``pattern_check.md`` - QOLU + BUIA first on an emphasised metric, top two
    otherwise, baselines in the published order - counted over the four
    scenarios x three metrics of one stage-2 window.
    """
    reference = str(summary.get("reference_method", REFERENCE_METHOD))
    matched = 0
    total = 0
    cells: list[dict[str, Any]] = []
    for configuration in summary.get("configurations", []):
        for scenario_block in configuration.get("scenarios", []):
            scenario = int(scenario_block["scenario"])
            for metric in PUBLISHED_METRICS:
                ranking = (scenario_block.get("ranking") or {}).get(metric)
                if not ranking:
                    continue
                total += 1
                checked = check_cell(scenario, metric, ranking, reference)
                matched += int(checked.agrees)
                cells.append(checked.as_record())
    return matched, total, cells
