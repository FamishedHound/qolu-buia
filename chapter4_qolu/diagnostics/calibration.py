"""Calibration of a categorical return distribution: randomised PIT and coverage.

The probability integral transform asks whether the realised value falls where
the forecast said it would: if ``Z`` is calibrated then ``u = F(y)`` is uniform
on ``[0, 1]``.  A C51 head predicts a *discrete* law on 51 fixed atoms, and the
ordinary PIT of a discrete forecast is not uniform even when the forecast is
perfect - it can only take the values the CDF steps through.  The standard
repair is the **randomised** PIT of Czado, Gneiting and Held (2009),

    u = F(y-) + v * p(y),    v ~ U(0, 1),

which is exactly uniform under a correct discrete forecast.  The randomisation
is therefore not a convenience: without it the histogram below would be
unreadable and the Kolmogorov-Smirnov statistic meaningless.

Design note (putting a continuous observation on a discrete support): ``p(y)``
only exists when ``y`` is an atom, so the realised return is snapped to its
nearest atom first, ties going upward.  That is the same discretisation the C51
projection ``Phi`` applies to every training target
(:func:`chapter4_qolu.agent.distributional.project_categorical_distribution`),
so the diagnostic asks the network about the quantity it was actually trained
on.  Returns outside ``[V_min, V_max]`` are clamped onto the end atoms exactly
as ``Phi`` clamps them, and the share of observations that had to be clamped is
reported next to the statistic rather than hidden inside it.

Design note (the p-value): the asymptotic Kolmogorov distribution, summed as in
Press et al.'s ``probks``, so the package keeps its numpy-only dependency set
(the same reason :mod:`chapter4_qolu.reporting.stats` computes Student's t by
hand).  It is approximate for small samples; an evaluation rollout here has
thousands of appraisals, where the approximation is not the binding constraint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "COVERAGE_LEVELS",
    "atom_indices",
    "calibration_statistics",
    "interval_coverage",
    "kolmogorov_smirnov_uniform",
    "out_of_support_fraction",
    "pit_histogram",
    "predictive_quantile",
    "randomised_pit",
]

#: The central intervals reported: 50 % and 90 %.
COVERAGE_LEVELS: tuple[float, ...] = (0.5, 0.9)
#: Bins of the PIT histogram; ten is the usual choice and keeps each bin's
#: expected count large enough to read at the sample sizes here.
PIT_BINS: int = 10
#: Guard against a cumulative sum falling a float's width short of a quantile.
_QUANTILE_TOLERANCE: float = 1e-9


def _support(atoms: np.ndarray) -> np.ndarray:
    support = np.asarray(atoms, dtype=np.float64).reshape(-1)
    if support.size < 2:
        raise ValueError("an atom support needs at least two atoms")
    return support


def atom_indices(atoms: np.ndarray, realised: np.ndarray) -> np.ndarray:
    """Index of the atom each realised value snaps to, ties rounding up."""
    support = _support(atoms)
    observations = np.asarray(realised, dtype=np.float64).reshape(-1)
    edges = 0.5 * (support[:-1] + support[1:])
    return np.searchsorted(edges, observations, side="right").astype(np.int64)


def out_of_support_fraction(atoms: np.ndarray, realised: np.ndarray) -> float:
    """Share of realised returns outside ``[V_min, V_max]``, i.e. clamped."""
    support = _support(atoms)
    observations = np.asarray(realised, dtype=np.float64).reshape(-1)
    outside = (observations < support[0]) | (observations > support[-1])
    return float(outside.mean())


def randomised_pit(
    probabilities: np.ndarray,
    atoms: np.ndarray,
    realised: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """``u = F(y-) + v * p(y)`` with ``v ~ U(0, 1)``, one draw per observation."""
    values = np.asarray(probabilities, dtype=np.float64)
    support = _support(atoms)
    observations = np.asarray(realised, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != support.size:
        raise ValueError("probabilities must have shape (n, n_atoms)")
    if observations.size != values.shape[0]:
        raise ValueError("one realised return per predicted distribution is required")

    indices = atom_indices(support, observations)
    rows = np.arange(values.shape[0])
    cumulative = np.cumsum(values, axis=1)
    below = np.where(indices > 0, cumulative[rows, np.maximum(indices - 1, 0)], 0.0)
    mass = values[rows, indices]
    draws = rng.random(values.shape[0])
    return np.clip(below + draws * mass, 0.0, 1.0)


def kolmogorov_smirnov_uniform(values: np.ndarray) -> tuple[float, float]:
    """Two-sided one-sample KS statistic of ``values`` against ``U(0, 1)``."""
    sample = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    size = sample.size
    if size == 0:
        raise ValueError("the KS statistic needs at least one observation")
    ranks = np.arange(1, size + 1, dtype=np.float64)
    above = float((ranks / size - sample).max())
    below = float((sample - (ranks - 1.0) / size).max())
    statistic = max(above, below, 0.0)
    return statistic, _ks_p_value(statistic, size)


def _ks_p_value(statistic: float, size: int) -> float:
    """``Q_KS(lambda)``, the asymptotic significance of a KS statistic."""
    root = float(np.sqrt(size))
    lam = (root + 0.12 + 0.11 / root) * float(statistic)
    if lam <= 0.0:
        return 1.0
    factor = 2.0
    total = 0.0
    previous = 0.0
    for order in range(1, 101):
        term = factor * float(np.exp(-2.0 * (order * lam) ** 2))
        total += term
        if abs(term) <= 1e-3 * previous or abs(term) <= 1e-8 * abs(total):
            return float(min(1.0, max(0.0, total)))
        factor = -factor
        previous = abs(term)
    return 1.0


def pit_histogram(values: np.ndarray, bins: int = PIT_BINS) -> dict[str, Any]:
    """Histogram of PIT values; a calibrated forecast has unit density."""
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    if sample.size == 0:
        raise ValueError("a PIT histogram needs at least one observation")
    if bins < 1:
        raise ValueError("bins must be positive")
    counts, edges = np.histogram(sample, bins=bins, range=(0.0, 1.0))
    width = 1.0 / bins
    return {
        "bins": int(bins),
        "edges": [float(edge) for edge in edges],
        "counts": [int(count) for count in counts],
        "density": [float(count) / (sample.size * width) for count in counts],
        "n": int(sample.size),
    }


def predictive_quantile(
    probabilities: np.ndarray, atoms: np.ndarray, quantile: float
) -> np.ndarray:
    """Generalised inverse CDF: the smallest atom with ``F(z) >= quantile``."""
    values = np.asarray(probabilities, dtype=np.float64)
    support = _support(atoms)
    if values.ndim != 2 or values.shape[1] != support.size:
        raise ValueError("probabilities must have shape (n, n_atoms)")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    cumulative = np.cumsum(values, axis=1)
    reached = cumulative >= float(quantile) - _QUANTILE_TOLERANCE
    indices = np.where(reached.any(axis=1), reached.argmax(axis=1), support.size - 1)
    return support[indices]


def interval_coverage(
    probabilities: np.ndarray,
    atoms: np.ndarray,
    realised: np.ndarray,
    levels: Sequence[float] = COVERAGE_LEVELS,
) -> list[dict[str, Any]]:
    """Empirical coverage of the central ``level`` intervals against nominal.

    Design note: the endpoints are atoms, because a discrete forecast has no
    others, so the interval is always at least as wide as the nominal one and
    an exactly calibrated model over-covers a little.  The mean width is
    reported alongside so that over-coverage from discreteness can be told from
    over-coverage from a forecast that is simply too vague.
    """
    observations = np.asarray(realised, dtype=np.float64).reshape(-1)
    rows: list[dict[str, Any]] = []
    for level in levels:
        if not 0.0 < float(level) < 1.0:
            raise ValueError("a coverage level must lie in (0, 1)")
        lower_q = (1.0 - float(level)) / 2.0
        upper_q = 1.0 - lower_q
        lower = predictive_quantile(probabilities, atoms, lower_q)
        upper = predictive_quantile(probabilities, atoms, upper_q)
        inside = (observations >= lower) & (observations <= upper)
        rows.append(
            {
                "level": float(level),
                "nominal": float(level),
                "empirical": float(inside.mean()),
                "mean_width": float((upper - lower).mean()),
                "lower_quantile": float(lower_q),
                "upper_quantile": float(upper_q),
                "n": int(observations.size),
            }
        )
    return rows


def calibration_statistics(
    probabilities: np.ndarray,
    atoms: np.ndarray,
    realised: np.ndarray,
    rng: np.random.Generator,
    levels: Sequence[float] = COVERAGE_LEVELS,
) -> dict[str, Any]:
    """The PIT block and the coverage block for one set of forecast pairs."""
    values = randomised_pit(probabilities, atoms, realised, rng)
    statistic, p_value = kolmogorov_smirnov_uniform(values)
    return {
        "pit": {
            "ks_statistic": statistic,
            "ks_p_value": p_value,
            "mean": float(values.mean()),
            "histogram": pit_histogram(values),
            "out_of_support_fraction": out_of_support_fraction(atoms, realised),
            "randomised": True,
        },
        "coverage": interval_coverage(probabilities, atoms, realised, levels),
    }
