"""CRPS of a categorical return distribution, and its skill against references.

The continuous ranked probability score

    CRPS(F, y) = integral (F(x) - 1{x >= y})^2 dx

is the proper score that answers "how good is this *distribution*", as opposed
to "how good is its mean".  For any ``F`` it equals the energy form

    CRPS(F, y) = E|X - y| - 1/2 E|X - X'|,    X, X' iid ~ F,

which for a 51-atom categorical law is two small sums, is exact rather than
quadrature, and is what :func:`crps` computes.

A CRPS on its own is not interpretable - it carries the units of the reward and
the scale of the run.  A **skill score** ``1 - CRPS_model / CRPS_reference`` is,
and the choice of reference is the whole argument:

*the network at initialisation*
    an untrained head with the same architecture and support.  Positive skill
    means training moved the distribution somewhere useful at all.

*climatology*
    a state-blind forecaster that always emits the run's own marginal histogram
    of realised returns.  It is the strongest reference that uses no state
    information whatsoever, so **positive skill against climatology is the
    evidence that** ``Z(s, a)`` **is state-conditional** - that the network has
    learnt an internal model of the return rather than memorising one
    distribution.  This is the headline number of the diagnostics.

Design note (climatology is fitted on the data it is scored on): the marginal
histogram is built from the same realised returns the skill score is then
computed over, which makes the reference slightly *better* than a climatology
estimated out of sample would be, and therefore makes the skill score
conservative.  An optimistic reference is the safe direction for a claim of the
form "the model beats it".
"""

from __future__ import annotations

import numpy as np

from chapter4_qolu.diagnostics.calibration import atom_indices

__all__ = [
    "climatology_crps",
    "climatology_probabilities",
    "crps",
    "crps_skill_score",
]


def crps(
    probabilities: np.ndarray, atoms: np.ndarray, realised: np.ndarray
) -> np.ndarray:
    """Per-observation CRPS of a categorical forecast against ``realised``."""
    values = np.asarray(probabilities, dtype=np.float64)
    support = np.asarray(atoms, dtype=np.float64).reshape(-1)
    observations = np.asarray(realised, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != support.size:
        raise ValueError("probabilities must have shape (n, n_atoms)")
    if observations.size != values.shape[0]:
        raise ValueError("one realised return per predicted distribution is required")

    absolute = np.abs(support[None, :] - observations[:, None])
    expected_error = (values * absolute).sum(axis=1)
    spread_matrix = np.abs(support[:, None] - support[None, :])
    expected_spread = np.einsum("ij,jk,ik->i", values, spread_matrix, values)
    return expected_error - 0.5 * expected_spread


def climatology_probabilities(atoms: np.ndarray, realised: np.ndarray) -> np.ndarray:
    """The run's marginal return histogram, binned onto the atom support."""
    support = np.asarray(atoms, dtype=np.float64).reshape(-1)
    observations = np.asarray(realised, dtype=np.float64).reshape(-1)
    if observations.size == 0:
        raise ValueError("a climatology needs at least one realised return")
    counts = np.bincount(
        atom_indices(support, observations), minlength=support.size
    ).astype(np.float64)
    return counts / counts.sum()


def climatology_crps(atoms: np.ndarray, realised: np.ndarray) -> np.ndarray:
    """Per-observation CRPS of the state-blind marginal forecaster."""
    support = np.asarray(atoms, dtype=np.float64).reshape(-1)
    observations = np.asarray(realised, dtype=np.float64).reshape(-1)
    row = climatology_probabilities(support, observations)
    tiled = np.repeat(row[None, :], observations.size, axis=0)
    return crps(tiled, support, observations)


def crps_skill_score(model: float, reference: float) -> float:
    """``1 - CRPS_model / CRPS_reference``; ``nan`` when the reference is 0.

    A reference score of zero means the reference forecast was perfect, and no
    skill score is defined against it.  Reporting ``nan`` keeps that visible
    rather than dividing by zero or silently returning "no skill".
    """
    denominator = float(reference)
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return 1.0 - float(model) / denominator
