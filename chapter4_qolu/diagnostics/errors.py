"""Point-error and support statistics of a predicted return distribution.

Two questions live here.

**How wrong is the number that actually drives decisions?**  The policy reads
``Z(s, a)`` only through ``E[Z]`` (see
:mod:`chapter4_qolu.agent.qolu_agent`), so the mean absolute error and the root
mean squared error of ``E[Z]`` against the realised return sit beside every
distributional statistic as the thing that would change behaviour if it moved.

**Is the atom support right for the signal?**  There are two ways for it to be
wrong and this module reports both.

*Too narrow.*  C51 clamps every target onto ``[V_min, V_max]``, so a return the
support cannot reach piles up on the two extreme atoms.  Mass parked there is
the standard diagnosis of a mis-specified support - Bellemare et al. (2017)
choose ``V`` from the reward scale for exactly this reason - and
:func:`boundary_mass` measures it.

*Too wide.*  The published support is 51 atoms on ``[-200, 200]``, i.e. an atom
spacing of ``delta_z = 8``.  If the realised returns of a run vary by much less
than one atom spacing, every one of them falls into the same atom cell and the
categorical head cannot express the variation at all, however well trained it
is.  That failure shows up as *no* boundary mass, so :func:`boundary_mass`
would call the support healthy; :func:`support_resolution` is what catches it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chapter4_qolu.diagnostics.calibration import atom_indices

__all__ = [
    "BOUNDARY_MASS_THRESHOLD",
    "boundary_mass",
    "error_statistics",
    "predicted_means",
    "support_resolution",
    "target_clustering",
]

#: Design note: two realised returns inside one iteration that differ by less
#: than this are the same number with floating-point noise on it, not a signal.
#: The ``objective`` and ``team`` readings pay one scalar per synchronised
#: update, so the difference being tested is exactly zero up to the arithmetic.
_CONSTANT_TOLERANCE: float = 1e-12

#: Design note: the level above which boundary mass is called out.  There is no
#: published threshold.  A distribution spread uniformly over ``N`` atoms puts
#: ``2 / N`` of its mass on the two extreme ones - 0.039 at the paper's 51
#: atoms - so 0.05 is the smallest round number that cannot be explained by the
#: support simply being wide.  The raw number is always reported as well, so a
#: reader who prefers another threshold can apply it.
BOUNDARY_MASS_THRESHOLD: float = 0.05


def _as_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("probabilities must have shape (n, n_atoms)")
    if values.shape[0] == 0:
        raise ValueError("probabilities must hold at least one row")
    return values


def _aligned(probabilities: np.ndarray, atoms: np.ndarray, realised: np.ndarray):
    """Validate and coerce the ``(p, z, y)`` triple every statistic takes."""
    values = _as_probabilities(probabilities)
    support = np.asarray(atoms, dtype=np.float64).reshape(-1)
    observations = np.asarray(realised, dtype=np.float64).reshape(-1)
    if support.size != values.shape[1]:
        raise ValueError(
            f"{values.shape[1]} atom probabilities against {support.size} atoms"
        )
    if observations.size != values.shape[0]:
        raise ValueError(
            f"{values.shape[0]} predictions against {observations.size} observations"
        )
    return values, support, observations


def predicted_means(probabilities: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    """``E[Z] = sum_i z_i p_i`` - the quantity the policy actually acts on."""
    values = _as_probabilities(probabilities)
    support = np.asarray(atoms, dtype=np.float64).reshape(-1)
    if support.size != values.shape[1]:
        raise ValueError(
            f"{values.shape[1]} atom probabilities against {support.size} atoms"
        )
    return values @ support


def error_statistics(
    probabilities: np.ndarray, atoms: np.ndarray, realised: np.ndarray
) -> dict[str, Any]:
    """MAE, RMSE and bias of ``E[Z]`` against the realised return."""
    values, support, observations = _aligned(probabilities, atoms, realised)
    predicted = values @ support
    residual = predicted - observations
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt((residual**2).mean())),
        "bias": float(residual.mean()),
        "mean_predicted": float(predicted.mean()),
        "mean_realised": float(observations.mean()),
        "sd_realised": float(observations.std()),
        "n": int(observations.size),
    }


def boundary_mass(
    probabilities: np.ndarray, threshold: float = BOUNDARY_MASS_THRESHOLD
) -> dict[str, Any]:
    """How much probability is parked on atoms ``0`` and ``N - 1``."""
    values = _as_probabilities(probabilities)
    lower = values[:, 0]
    upper = values[:, -1]
    total = lower + upper
    return {
        "lower_mean": float(lower.mean()),
        "upper_mean": float(upper.mean()),
        "total_mean": float(total.mean()),
        "total_max": float(total.max()),
        "threshold": float(threshold),
        "fraction_above_threshold": float((total > threshold).mean()),
        "flagged": bool(total.mean() > threshold),
        "n": int(values.shape[0]),
    }


def target_clustering(
    iteration_index: np.ndarray, realised: np.ndarray
) -> dict[str, Any]:
    """How many *independent* observations the statistics above really have.

    One iteration produces one appraisal per candidate - 50 of them at the
    published ``samples_per_iteration`` - and under the ``objective`` and
    ``team`` readings of ``R_QOLU,i`` the stakeholder is paid **one scalar for
    the whole synchronised update** (see
    :func:`chapter4_qolu.runner._transition_rewards`).  Every appraisal in that
    iteration is then scored against the same realised return, so the nominal
    sample size of ``iterations * samples_per_iteration`` overstates the
    independent information by a factor of ``samples_per_iteration``.

    Nothing above is wrong because of it - an MAE, a CRPS and a coverage
    fraction are all still exactly what they say - but a *p-value* computed
    from the nominal count is not, so the effective count is reported beside
    them rather than left for a reader to work out.  Under ``per_parcel`` the
    reward varies within an iteration and the effective count is the nominal
    one.
    """
    iterations = np.asarray(iteration_index).reshape(-1)
    observations = np.asarray(realised, dtype=np.float64).reshape(-1)
    if iterations.size != observations.size:
        raise ValueError("one iteration index per realised return is required")
    if observations.size == 0:
        raise ValueError("clustering needs at least one appraisal")

    unique = np.unique(iterations)
    constant = True
    for value in unique:
        block = observations[iterations == value]
        if block.size > 1 and float(block.max() - block.min()) > _CONSTANT_TOLERANCE:
            constant = False
            break
    n_iterations = int(unique.size)
    return {
        "n_appraisals": int(observations.size),
        "n_iterations": n_iterations,
        "appraisals_per_iteration": float(observations.size) / n_iterations,
        "distinct_realised_values": int(np.unique(observations).size),
        "constant_within_iteration": bool(constant),
        "effective_n": n_iterations if constant else int(observations.size),
    }


def support_resolution(atoms: np.ndarray, realised: np.ndarray) -> dict[str, Any]:
    """Can the atom spacing express the variation in the realised return?

    Design note (the test): the support is said to resolve the signal when the
    standard deviation of the realised return is at least one atom spacing.
    Below that, a run's returns occupy one or two atom cells and every
    distributional statistic is measuring the discretisation rather than the
    model, which has to be said out loud rather than left for a reader to
    notice from ``distinct_atoms_used``.
    """
    support = np.asarray(atoms, dtype=np.float64).reshape(-1)
    if support.size < 2:
        raise ValueError("an atom support needs at least two atoms")
    observations = np.asarray(realised, dtype=np.float64).reshape(-1)
    if observations.size == 0:
        raise ValueError("support resolution needs at least one realised return")
    delta_z = (float(support[-1]) - float(support[0])) / (support.size - 1)
    deviation = float(observations.std())
    return {
        "n_atoms": int(support.size),
        "v_min": float(support[0]),
        "v_max": float(support[-1]),
        "delta_z": delta_z,
        "realised_sd": deviation,
        "realised_sd_in_atoms": deviation / delta_z if delta_z > 0 else float("inf"),
        "realised_range": float(observations.max() - observations.min()),
        "distinct_atoms_used": int(np.unique(atom_indices(support, observations)).size),
        "resolves_the_signal": bool(deviation >= delta_z),
    }
