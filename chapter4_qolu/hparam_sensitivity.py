"""The +/-20 % hyper-parameter perturbations described in Chapter 1.

Chapter 1 states::

    Sensitivity analyses are conducted by perturbing key hyperparameters
    (typically +/-20%) to assess the robustness of comparative conclusions.

This module supplies the arithmetic behind that analysis: what +/-20 % of a
hyper-parameter is, and what has to happen when 20 % of a value is not
expressible - an integer that stops being an integer, an odd window size that
stops being odd, a probability that would leave [0, 1].  Every such adjustment is recorded on the :class:`Perturbation`
rather than applied silently, because the report has to be able to say *what
was actually run*, not what was asked for.

*Which* hyper-parameters are in scope, and why, lives next door in
:mod:`chapter4_qolu.hparam_catalogue`; its names are re-exported here so that
one import reaches the whole vocabulary.

The claim under test is about **comparative conclusions**, i.e. the ordering of
the four methods, so the consumer of this module
(:mod:`chapter4_qolu.run_hparam_sensitivity`) compares rankings and Kendall's
tau rather than absolute values.

Nothing here imports torch, loads a raster or builds an environment, so the
list of perturbations and their cost can be inspected instantly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from chapter4_qolu.hparam_catalogue import (
    KEY_HYPERPARAMETERS,
    HyperParameter,
    format_value,
    hyperparameter_names,
    resolve_hyperparameters,
)

__all__ = [
    "KEY_HYPERPARAMETERS",
    "PERTURBATION_FRACTION",
    "HyperParameter",
    "Perturbation",
    "format_value",
    "hyperparameter_names",
    "kendall_tau",
    "perturb",
    "perturbations",
    "resolve_hyperparameters",
]

#: Design note (the fraction): 0.20, the "typically +/-20%" of the sentence
#: quoted above.  It is the *default*, not a constant of the analysis - the
#: driver exposes ``--fraction`` so the same machinery can answer "and at 10 %?"
#: without a second implementation.
PERTURBATION_FRACTION: float = 0.20


def _round_half_away_from_zero(value: float) -> int:
    """Round to the nearest integer, ties away from zero.

    Design note: not :func:`round`, which is banker's rounding - 0.5 would go
    to 0 and 1.5 to 2, so two perturbations of the same shape would round in
    opposite directions and the report could not state one rule.
    """
    if value >= 0.0:
        return math.floor(value + 0.5)
    return -math.floor(-value + 0.5)


def _make_odd(value: int, direction: int) -> int:
    """Nudge an even integer onto the next odd one, away from the baseline."""
    return value if value % 2 == 1 else value + (1 if direction > 0 else -1)


@dataclass(frozen=True)
class Perturbation:
    """One hyper-parameter moved one way, and every adjustment that took."""

    hyperparameter: HyperParameter
    #: ``+1`` or ``-1``.
    direction: int
    fraction: float
    #: The value +/-``fraction`` asks for, before rounding or clamping.
    requested: float
    #: The value actually configured.
    value: float | int
    rounded: bool
    clamped: bool

    @property
    def name(self) -> str:
        return self.hyperparameter.name

    @property
    def direction_label(self) -> str:
        return "plus" if self.direction > 0 else "minus"

    @property
    def label(self) -> str:
        """Filesystem-safe name, e.g. ``learning_rate_plus``."""
        return f"{self.name}_{self.direction_label}"

    @property
    def description(self) -> str:
        sign = "+" if self.direction > 0 else "-"
        return (
            f"`{self.name}` {sign}{self.fraction * 100:.0f} %: "
            f"{format_value(self.hyperparameter.baseline)} -> "
            f"{format_value(self.value)}"
        )

    @property
    def target(self) -> str:
        return self.hyperparameter.target

    @property
    def is_no_op(self) -> bool:
        """Whether rounding or clamping put the value back on the baseline.

        Design note: such a perturbation is *listed* - hiding it would make the
        analysis look more complete than it is - but it is never run and never
        counted, because "the ranking survived" would then be true by
        construction rather than by experiment.
        """
        return float(self.value) == float(self.hyperparameter.baseline)

    @property
    def applied_fraction(self) -> float:
        """The relative move that was actually made, after every adjustment."""
        baseline = float(self.hyperparameter.baseline)
        if baseline == 0.0:  # pragma: no cover - no key baseline is zero
            return 0.0
        return (float(self.value) - baseline) / baseline

    def overrides(self) -> dict[str, float | int]:
        """The single configuration field this perturbation sets."""
        return {self.name: self.value}

    def as_record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "hyperparameter": self.name,
            "direction": self.direction_label,
            "target": self.target,
            "source": self.hyperparameter.source,
            "note": self.hyperparameter.note,
            "fraction": self.fraction,
            "baseline": self.hyperparameter.baseline,
            "baseline_text": format_value(self.hyperparameter.baseline),
            "requested": self.requested,
            "requested_text": format_value(self.requested),
            "value": self.value,
            "value_text": format_value(self.value),
            "applied_fraction": self.applied_fraction,
            "rounded": bool(self.rounded),
            "clamped": bool(self.clamped),
            "no_op": bool(self.is_no_op),
            "description": self.description,
        }


def perturb(
    hyperparameter: HyperParameter,
    direction: int,
    fraction: float = PERTURBATION_FRACTION,
) -> Perturbation:
    """Move one hyper-parameter by ``fraction`` in ``direction``.

    Integer fields are rounded half away from zero, odd ones are then nudged
    away from the baseline, and bounded ones are clamped.  Each of the three is
    flagged on the result, so the report can print the requested value beside
    the configured one instead of quietly presenting the second as the first.
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +1 or -1, got {direction!r}")
    if fraction <= 0.0:
        raise ValueError("fraction must be positive")

    item = hyperparameter
    requested = float(item.baseline) * (1.0 + direction * float(fraction))
    value: float | int = requested
    rounded = False
    if item.integral:
        value = _round_half_away_from_zero(requested)
        if item.odd:
            value = _make_odd(int(value), direction)
        rounded = float(value) != requested

    clamped = False
    if item.minimum is not None and float(value) < float(item.minimum):
        value = int(item.minimum) if item.integral else float(item.minimum)
        clamped = True
    if item.maximum is not None and float(value) > float(item.maximum):
        value = int(item.maximum) if item.integral else float(item.maximum)
        clamped = True

    return Perturbation(
        hyperparameter=item,
        direction=int(direction),
        fraction=float(fraction),
        requested=requested,
        value=value,
        rounded=rounded,
        clamped=clamped,
    )


def perturbations(
    hyperparameters: Sequence[HyperParameter],
    fraction: float = PERTURBATION_FRACTION,
) -> tuple[Perturbation, ...]:
    """Both directions of every hyper-parameter, minus first, in order."""
    return tuple(
        perturb(item, direction, fraction)
        for item in hyperparameters
        for direction in (-1, 1)
    )


def kendall_tau(first: Sequence[str], second: Sequence[str]) -> float:
    """Kendall's tau between two orderings of the same methods.

    Design note (tau-a, not tau-b): a ranking produced by
    :func:`chapter4_qolu.reporting.aggregate.rank_methods` is a **total** order
    - exact ties are broken by method name so that a summary is byte-for-byte
    reproducible - so neither input can contain a tie and the simple
    ``(concordant - discordant) / (n (n - 1) / 2)`` form is exact.  A single
    method has no pairs at all; it is reported as 1.0, the value that says "the
    two orderings agree", rather than as a division by zero.
    """
    left = list(first)
    right = list(second)
    if sorted(left) != sorted(right):
        raise ValueError(
            "Kendall's tau needs the same methods in both rankings, got "
            f"{sorted(left)} and {sorted(right)}"
        )
    n = len(left)
    if n < 2:
        return 1.0
    position = {name: index for index, name in enumerate(right)}
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if position[left[i]] < position[left[j]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)
