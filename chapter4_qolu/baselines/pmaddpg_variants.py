"""The three readings of the P-MADDPG baseline, and their random streams.

Split out of :mod:`chapter4_qolu.baselines.pmaddpg` so that the *method* keeps
holding nothing but Chapter 3's Algorithm 1 while "which reading is this"
and "where do its random numbers come from" live in one small place.

The variant axis
----------------

======================  ==========================================================
``published``           Chapter 3's own hyper-parameters (lr 1e-4, ``gamma`` 0.1,
                        hard target copy every 250 updates, epsilon-greedy 0.3,
                        the linearisation constant) **plus** the Lowe et al.
                        policy regulariser that keeps the sigmoid off its tails.
``v1``                  The unregularised reading: Adam 1e-3, ``gamma`` 0.99,
                        Polyak ``tau`` 0.01, Ornstein-Uhlenbeck exploration,
                        ``alpha`` = 0 and *no* policy regulariser.
``reg_only``            ``v1`` plus the regulariser and nothing else, so the
                        effect of the regulariser can be separated from the
                        effect of Chapter 3's published numbers.
======================  ==========================================================

Design note (what ``v1`` selects): the *settings*, not a particular random
stream.  The seeding described below is shared by every variant.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = [
    "PMADDPG_VARIANTS",
    "SeedStreams",
    "seed_streams",
    "variant_settings",
]

#: Named departures from :class:`~chapter4_qolu.baselines.config.PMADDPGConfig`.
#: ``published`` is empty because the class defaults *are* Chapter 3's numbers.
PMADDPG_VARIANTS: Mapping[str, Mapping[str, object]] = {
    "published": {},
    "v1": {
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 1e-3,
        "gamma": 0.99,
        "tau": 0.01,
        "target_update_interval": None,
        "epsilon_greedy": None,
        "policy_regulariser": 0.0,
        "team_reward_weight": 0.0,
    },
    "reg_only": {
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 1e-3,
        "gamma": 0.99,
        "tau": 0.01,
        "target_update_interval": None,
        "epsilon_greedy": None,
        "policy_regulariser": 1e-3,
        "team_reward_weight": 0.0,
    },
}


def variant_settings(variant: str) -> dict[str, object]:
    """The configuration overrides named by ``variant`` (a fresh copy)."""
    try:
        overrides = PMADDPG_VARIANTS[str(variant)]
    except KeyError:
        raise ValueError(
            f"unknown P-MADDPG variant {variant!r}; expected one of "
            f"{tuple(sorted(PMADDPG_VARIANTS))}"
        ) from None
    return dict(overrides)


class SeedStreams:
    """Statistically independent random streams for one P-MADDPG run.

    Deriving everything from ``seed + index`` - agent ``i`` of run ``s``
    initialised from ``s + i`` and its noise process from the same number -
    would make runs ``s`` and ``s + 1`` share two of their three network
    initialisations *and* two of their three noise streams, so the
    replicates of a campaign cell would not be independent.

    :class:`numpy.random.SeedSequence` exists for exactly this: spawning splits
    one seed into children that are independent by construction, so nothing has
    to be offset by hand and no two runs can collide.  Four named children are
    spawned - replay sampling, network initialisation, exploration at
    construction and exploration at :meth:`reset` - and each of the last three
    is spawned once more, per agent.
    """

    #: Child order of the root sequence.  Fixed, because it is what makes a run
    #: reproducible from its seed.
    ROLES: tuple[str, ...] = ("replay", "initialisation", "noise", "reset")

    def __init__(self, seed: int | None, n_agents: int) -> None:
        if n_agents < 1:
            raise ValueError("n_agents must be positive")
        self.seed = seed
        self.n_agents = int(n_agents)
        children = np.random.SeedSequence(seed).spawn(len(self.ROLES))
        self._children = dict(zip(self.ROLES, children))

    def generator(self, role: str) -> np.random.Generator:
        """One generator for a whole-run role, e.g. replay sampling."""
        return np.random.default_rng(self._child(role))

    def agent_seeds(self, role: str) -> tuple[int, ...]:
        """One 32-bit seed per agent, for seeding torch locally."""
        return tuple(
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in self._child(role).spawn(self.n_agents)
        )

    def agent_generators(self, role: str) -> tuple[np.random.Generator, ...]:
        """One generator per agent, e.g. its exploration noise."""
        return tuple(
            np.random.default_rng(child)
            for child in self._child(role).spawn(self.n_agents)
        )

    def _child(self, role: str) -> np.random.SeedSequence:
        try:
            return self._children[role]
        except KeyError:
            raise ValueError(
                f"unknown seed role {role!r}; expected one of {self.ROLES}"
            ) from None


def seed_streams(seed: int | None, n_agents: int) -> SeedStreams:
    """Independent random streams for a run of ``n_agents`` agents."""
    return SeedStreams(seed, n_agents)
