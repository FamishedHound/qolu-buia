"""The method registry: which methods exist, and which are variants of which.

Split out of :mod:`chapter4_qolu.scenario_worker` so that the registry can be
imported without torch, a raster or an environment - the reporting modules, the
window scan and the sensitivity sweep all need to know the method *names* long
before anything is executed.  ``scenario_worker`` re-exports
:data:`METHOD_KEYS` and :data:`QOLU_METHOD`.

Six of the keys are **variants** of a published method rather than methods in
their own right:

======================  =============  =====================================
key                     published as   what it varies
======================  =============  =====================================
``pmaddpg_v1``          P-MADDPG       neither Chapter 3's hyper-parameters
                                       nor the policy regulariser
``pmaddpg_reg_only``    P-MADDPG       the policy regulariser, alone
``fixed_ltos``          LToS           sharing frozen at the initialiser
``mitm_learned``        MitM           stakeholders nudge ``theta_m``
``qolu_buia_per``       QOLU + BUIA    prioritised replay, a Rainbow extra
                                       the chapter never mentions
``qolu_c51``            QOLU + BUIA    Algorithm 1 without the NoisyNet
                                       layers of line 5
======================  =============  =====================================

One key is an **ablation** with no published counterpart: ``no_regulator``,
the bottom-up investor alone - BUIA proposes and every
proposal is converted, with no top-down appraisal.  It is its own canonical
form and is left out of every comparison with the published tables.

Design note (variant naming): a variant keeps its own key everywhere a *run* is
identified - the result path, the run record, the campaign tables - so two
variants can never overwrite each other, and is mapped onto its published
parent only when a result is compared with the published Tables 4-7, where the
paper has exactly one row per method.  :func:`canonical_method` is that map,
and it is the only place the mapping is made.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "METHOD_KEYS",
    "NO_REGULATOR_METHOD",
    "PMADDPG_METHOD_VARIANTS",
    "PUBLISHED_METHODS",
    "QOLU_METHOD",
    "QOLU_METHOD_VARIANTS",
    "VARIANT_PARENTS",
    "canonical_method",
    "is_variant",
]

#: The published system: BUIA proposes, the QOLU stakeholders appraise.
QOLU_METHOD = "qolu_buia"

#: The ablation: BUIA proposes and every proposal is converted.
NO_REGULATOR_METHOD = "no_regulator"

#: Every method key a campaign accepts.  ``qolu_buia`` is the published system;
#: the rest are its two variants, the baselines in
#: :mod:`chapter4_qolu.baselines` with their variants, and the ``no_regulator``
#: ablation.
METHOD_KEYS: tuple[str, ...] = (
    QOLU_METHOD,
    "qolu_buia_per",
    "qolu_c51",
    "pmaddpg",
    "pmaddpg_v1",
    "pmaddpg_reg_only",
    "ltos",
    "fixed_ltos",
    "mitm",
    "mitm_learned",
    NO_REGULATOR_METHOD,
)

#: A variant key mapped onto the method the published tables have a row for.
VARIANT_PARENTS: Mapping[str, str] = {
    "pmaddpg_v1": "pmaddpg",
    "pmaddpg_reg_only": "pmaddpg",
    "fixed_ltos": "ltos",
    "mitm_learned": "mitm",
    "qolu_buia_per": QOLU_METHOD,
    "qolu_c51": QOLU_METHOD,
}

#: Campaign method key -> the ``variant`` name
#: :func:`chapter4_qolu.agent.qolu_variants.qolu_variant_settings` takes.
#:
#: Design note (QOLU variant axis): QOLU combines the Rainbow components with a
#: C51 head, but the chapter states only part of that.  Algorithm 1 line 5 *does*
#: say actions are selected "by exploiting noisy network parameters theta", so
#: the NoisyNet layers are in the published reading and ``qolu_buia`` carries
#: them.  Prioritised replay is a Rainbow component the chapter never mentions
#: - Algorithm 1 says only "sample a minibatch of transitions from D" - so it
#: gets a key of its own, ``qolu_buia_per``, and stays off by default.
#: ``qolu_c51`` carries neither.
QOLU_METHOD_VARIANTS: Mapping[str, str] = {
    QOLU_METHOD: "published",
    "qolu_buia_per": "per",
    "qolu_c51": "c51",
}

#: Campaign method key -> the ``variant`` name
#: :meth:`chapter4_qolu.baselines.pmaddpg.PMADDPGMethod.for_env` takes.
#:
#: Design note (P-MADDPG variant axis): taken literally, Chapter 3's actor loss
#: is unbounded in the pre-activation, and with the squashed actor this
#: environment needs it is seed-bistable - on some seeds the sigmoid saturates
#: and the method stops approving anything.  Three readings are therefore
#: carried side by side: ``published`` (the default: Lowe et al.'s policy
#: regulariser plus Chapter 3's published hyper-parameters), ``v1`` (neither)
#: and ``reg_only`` (the regulariser alone, so its effect can be separated from
#: that of the published hyper-parameters).  ``mask_terminal`` is not swept: it
#: is a no-op on this environment, which has no terminal state inside a run.
PMADDPG_METHOD_VARIANTS: Mapping[str, str] = {
    "pmaddpg": "published",
    "pmaddpg_v1": "v1",
    "pmaddpg_reg_only": "reg_only",
}

#: The four rows of the published Tables 4-7.
PUBLISHED_METHODS: tuple[str, ...] = (QOLU_METHOD, "pmaddpg", "ltos", "mitm")


def is_variant(method: str) -> bool:
    """``True`` when ``method`` is a variant of a published method."""
    return str(method) in VARIANT_PARENTS


def canonical_method(method: str) -> str:
    """The published method ``method`` stands in for (itself, if published)."""
    name = str(method)
    return VARIANT_PARENTS.get(name, name)
