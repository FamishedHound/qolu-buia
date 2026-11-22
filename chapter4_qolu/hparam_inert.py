"""Telling "the ranking survived" apart from "nothing happened".

A +/-20 % move is only evidence about robustness if it *reached the run*.  Two
of the eleven key hyper-parameters cannot always do so at a 1000-iteration
budget, and the arithmetic is not subtle:

* a run stores ``samples_per_iteration x iterations`` transitions per
  stakeholder - 50 000 at the published settings - so a ``replay_capacity`` of
  80 000 or 120 000 never evicts anything, and both directions leave the buffer
  contents identical;
* ``learn()`` is called once per iteration and only after ``learning_starts``
  transitions have accumulated, so a run takes about 980 gradient steps and any
  ``target_sync_interval`` above that - 1 000 or 1 200, say - fires the hard
  reset the same number of times: zero.

:func:`chapter4_qolu.hparam_sensitivity.perturb` already reports the
perturbations that are the identity *arithmetically* (``is_no_op``).  These are
different: the value really did change, the run really was performed, and it
came back **bit-identical** to the baseline's.  Scoring that as "the ranking
was preserved" would be true by construction rather than by experiment, so the
payload and the report separate the two and the headline counts only the
perturbations that moved something.

The predicate is deliberately **exact**.  Two runs of the same seed, crop and
settings are deterministic on CPU torch, so a perturbation that could reach the
landscape at all moves some digit of some metric; requiring a tolerance would
only let a real - if tiny - effect be filed as no effect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["INERT_REASON", "inert_perturbation", "informative_perturbations"]

#: How the payload and the report explain an inert perturbation.  The word
#: "identical" is what the reader has to come away with: the runs were
#: performed and came back the same, so the row carries no evidence either way.
INERT_REASON = (
    "the perturbed value was configured and the runs were performed, but every "
    "re-run method came back with per-seed metrics identical to the baseline's, "
    "so this perturbation could not reach the landscape at this run length and "
    "its preserved ranking is a construction rather than a result"
)


def _compared_entries(
    scenarios: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Every method entry that is actually a comparison.

    A row borrowed from the baseline campaign equals it by definition (it *is*
    the baseline record), and a row with no baseline counterpart has nothing to
    be equal to; neither is evidence, so neither is counted.
    """
    entries: list[Mapping[str, Any]] = []
    for scenario in scenarios:
        for cell in (scenario.get("metrics") or {}).values():
            for entry in (cell.get("methods") or {}).values():
                if entry.get("borrowed_from_baseline"):
                    continue
                if entry.get("baseline_values") is None:
                    continue
                entries.append(entry)
    return entries


def inert_perturbation(scenarios: Sequence[Mapping[str, Any]]) -> bool:
    """Whether every re-run method reproduced the baseline exactly."""
    entries = _compared_entries(scenarios)
    if not entries:
        return False
    return all(
        list(entry.get("values") or []) == list(entry.get("baseline_values") or [])
        for entry in entries
    )


def informative_perturbations(
    blocks: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """The perturbations that were run *and* changed something."""
    return [
        block
        for block in blocks
        if not block.get("no_op") and block.get("scenarios") and not block.get("inert")
    ]
