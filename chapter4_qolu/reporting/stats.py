"""Dependency-free statistics for the campaign result tables.

The paper's Section 5 says only: "we conducted paired *t*-tests for each
scenario and metric to evaluate the confidence in our results. All scores are
presented normalized with standard errors."  No test statistic, no degrees of
freedom and no seed count are reported, so everything below is the standard
textbook construction rather than a reproduction of a published procedure.

Design note (no SciPy): SciPy happens to be importable in the interpreter this
package is developed against, but ``chapter4_qolu`` depends on nothing
beyond numpy and torch, so Student's *t* distribution is implemented
here instead of imported.  The two-sided p-value uses the exact identity

    P(|T_df| >= |t|) = I_{df / (df + t^2)}(df / 2, 1 / 2)

with ``I_x(a, b)`` the regularised incomplete beta function, evaluated by the
Lentz continued fraction (Numerical Recipes 6.4).  A test cross-checks the
result against ``scipy.stats.ttest_rel`` wherever SciPy is installed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "MIN_PAIRED_SAMPLES",
    "PairedTTest",
    "SampleSummary",
    "cohens_d_paired",
    "paired_t_test",
    "regularised_incomplete_beta",
    "summarise",
    "two_sided_t_p_value",
]

#: Design note: a paired t-test is only reported from three seeds upwards.  Two
#: pairs give ``df = 1``, where the t distribution has no finite variance and a
#: p-value carries no information; the campaign therefore reports "not
#: possible" rather than a number that would be quoted out of context.
MIN_PAIRED_SAMPLES: int = 3

_TINY = 1e-300
_BETACF_EPSILON = 1e-15
_BETACF_MAX_ITERATIONS = 300

#: Design note: paired differences whose standard deviation is this small
#: relative to their mean count as having *no* variance.  Subtracting two
#: metric values that agree to the last bit leaves a spread around 1e-17, and
#: dividing by it would report ``t = 1e16`` - a number made entirely of
#: floating-point noise.  Real metric spreads across seeds are of order 1e-3.
_ZERO_VARIANCE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SampleSummary:
    """Mean, sample standard deviation and standard error of the mean.

    ``values`` keeps the sample itself, in the order it was given (per-seed
    order for a campaign).  The published tables quote two decimals on a 0.05
    lattice; carrying the raw sample means ``summary.json`` can be re-rendered
    at any precision - or re-tested - without re-running anything.
    """

    n: int
    mean: float
    sd: float
    se: float
    values: tuple[float, ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "sd": self.sd,
            "se": self.se,
            "values": list(self.values),
        }


def summarise(values: Sequence[float]) -> SampleSummary:
    """Summarise a sample of per-seed values.

    The standard deviation uses ``ddof = 1`` (the sample estimator) and the
    standard error is ``sd / sqrt(n)``, which is what the published tables'
    "$\\pm$" column reports.  A single value has no spread, so both are zero.
    """
    sample = tuple(float(value) for value in values)
    array = np.asarray(sample, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarise an empty sample")
    n = int(array.size)
    mean = float(array.mean())
    if n < 2:
        return SampleSummary(n=n, mean=mean, sd=0.0, se=0.0, values=sample)
    sd = float(array.std(ddof=1))
    return SampleSummary(n=n, mean=mean, sd=sd, se=sd / math.sqrt(n), values=sample)


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz's algorithm for the continued fraction of ``I_x(a, b)``."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, _BETACF_MAX_ITERATIONS + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < _BETACF_EPSILON:
            break
    return h


def regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)``, the regularised incomplete beta function."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def two_sided_t_p_value(t: float, df: int) -> float:
    """``P(|T_df| >= |t|)`` for Student's t distribution."""
    if df < 1:
        raise ValueError("df must be at least one")
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    statistic = abs(float(t))
    x = df / (df + statistic * statistic)
    return float(regularised_incomplete_beta(df / 2.0, 0.5, x))


def _differences(treatment: Sequence[float], control: Sequence[float]) -> np.ndarray:
    first = np.asarray(list(treatment), dtype=np.float64)
    second = np.asarray(list(control), dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(
            f"paired samples must have the same length, got {first.size} and "
            f"{second.size}"
        )
    return first - second


def _has_no_variance(sd: float, mean_difference: float) -> bool:
    """Whether a spread is indistinguishable from floating-point noise."""
    return sd <= _ZERO_VARIANCE_TOLERANCE * max(abs(mean_difference), 1.0)


def cohens_d_paired(
    treatment: Sequence[float], control: Sequence[float]
) -> float | None:
    """Cohen's ``d_z`` for paired samples: ``mean(diff) / sd(diff)``.

    Design note: the paired (``d_z``) form is reported rather than the
    independent-samples ``d``, because the campaign pairs *the same seed* under
    two methods; ``None`` when the differences have no variance at all.
    """
    differences = _differences(treatment, control)
    if differences.size < 2:
        return None
    sd = float(differences.std(ddof=1))
    mean_difference = float(differences.mean())
    if _has_no_variance(sd, mean_difference):
        return 0.0 if mean_difference == 0.0 else None
    return float(mean_difference / sd)


@dataclass(frozen=True)
class PairedTTest:
    """Outcome of one paired t-test (or the reason there is none)."""

    n: int
    mean_difference: float
    t: float | None
    df: int | None
    p_value: float | None
    cohens_d: float | None
    possible: bool
    note: str

    @property
    def significant(self) -> bool:
        """``True`` when the two-sided p-value is below 0.05."""
        return self.p_value is not None and self.p_value < 0.05

    def as_record(self) -> dict[str, object]:
        return {
            "n": self.n,
            "mean_difference": self.mean_difference,
            "t": self.t,
            "df": self.df,
            "p_value": self.p_value,
            "cohens_d": self.cohens_d,
            "possible": self.possible,
            "significant": self.significant,
            "note": self.note,
        }


def paired_t_test(
    treatment: Sequence[float],
    control: Sequence[float],
    min_samples: int = MIN_PAIRED_SAMPLES,
) -> PairedTTest:
    """Paired (one-sample-on-the-differences) t-test of ``treatment - control``.

    The sign follows the direction of the difference, so a positive ``t`` means
    ``treatment`` scored higher.  Two degenerate cases are reported explicitly
    rather than raised:

    * fewer than ``min_samples`` pairs - no test at all;
    * zero variance in the differences - the statistic is ``0`` (identical
      samples, ``p = 1``) or infinite (a constant non-zero gap, ``p = 0``).
    """
    differences = _differences(treatment, control)
    n = int(differences.size)
    mean_difference = float(differences.mean()) if n else 0.0
    if n < max(2, int(min_samples)):
        return PairedTTest(
            n=n,
            mean_difference=mean_difference,
            t=None,
            df=None,
            p_value=None,
            cohens_d=None,
            possible=False,
            note=(
                f"n = {n} paired seed(s): no paired t-test is possible "
                f"(at least {min_samples} are needed)"
            ),
        )

    df = n - 1
    sd = float(differences.std(ddof=1))
    if _has_no_variance(sd, mean_difference):
        identical = mean_difference == 0.0
        return PairedTTest(
            n=n,
            mean_difference=mean_difference,
            t=0.0 if identical else math.copysign(math.inf, mean_difference),
            df=df,
            p_value=1.0 if identical else 0.0,
            cohens_d=0.0 if identical else None,
            possible=True,
            note=(
                "the paired differences have zero variance across seeds"
                + (" and are all zero" if identical else "")
            ),
        )

    standard_error = sd / math.sqrt(n)
    t = mean_difference / standard_error
    return PairedTTest(
        n=n,
        mean_difference=mean_difference,
        t=float(t),
        df=df,
        p_value=two_sided_t_p_value(t, df),
        cohens_d=float(mean_difference / sd),
        possible=True,
        note="",
    )
