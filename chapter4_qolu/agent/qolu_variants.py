"""The three readings of QOLU, and the ``QOLUConfig`` overrides each one needs.

Laid out like :mod:`chapter4_qolu.baselines.pmaddpg_variants`, for the same
reason: every reading of the published text stays runnable, so a campaign can
compare them instead of choosing between them in advance.

========================  ==================================================
``published``             What the chapter actually states: the C51 head,
                          n-step returns, double Q-learning, the dueling
                          streams, epsilon-greedy **and** the NoisyNet layers
                          of Algorithm 1 line 5 ("exploiting noisy network
                          parameters theta"), over a *uniform* replay buffer -
                          the only sampling rule Algorithm 1 describes.  The
                          default, and the ``qolu_buia`` method key.
``per``                   ``published`` plus prioritised experience replay.
                          PER is a Rainbow component (Hessel et al. 2018),
                          but the chapter never mentions prioritisation, so
                          it is an opt-in extra rather than a default: the
                          ``qolu_buia_per`` method key, so that the two
                          readings can be compared.
``c51``                   Algorithm 1 with ordinary linear layers as well as a
                          uniform buffer, so the effect of the NoisyNet layers
                          can be measured on its own.  The ``qolu_c51`` method
                          key.
========================  ==================================================

Design note (what ``c51`` selects): the *settings*.  The switches select the
plain layer path rather than approximating it, so a ``c51`` run is the
non-noisy agent exactly, not an approximation of it.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["QOLU_VARIANTS", "qolu_variant_settings"]

#: Named departures from :class:`chapter4_qolu.config.QOLUConfig`.
#: ``published`` is empty because the class defaults *are* the chapter's
#: reading: NoisyNet layers on, prioritised replay off.
QOLU_VARIANTS: Mapping[str, Mapping[str, object]] = {
    "published": {},
    "per": {"prioritised_replay": True},
    "c51": {"noisy_nets": False, "prioritised_replay": False},
}


def qolu_variant_settings(variant: str) -> dict[str, object]:
    """The configuration overrides named by ``variant`` (a fresh copy)."""
    try:
        overrides = QOLU_VARIANTS[str(variant)]
    except KeyError:
        raise ValueError(
            f"unknown QOLU variant {variant!r}; expected one of "
            f"{tuple(sorted(QOLU_VARIANTS))}"
        ) from None
    return dict(overrides)
