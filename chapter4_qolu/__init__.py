"""Chapter 4 (IJCAI 2025): urban land-use POSG + QOLU agent.

This package implements the environment and the agents described in:

    Pelcner, Marcolino, do Carmo Alves, Harrison & Atkinson (2025),
    "Multi-Objective Quantile-Based Reinforcement Learning for Modern Urban
    Planning", IJCAI 2025.

The implementation follows the published Algorithm 1, Table 1 and Section 3.

Naming caveat (see :mod:`chapter4_qolu.agent.distributional` for the full
discussion): the paper's *title and prose* speak of quantile regression, while
its *pseudocode and hyper-parameter table* describe a categorical, fixed-atom
distributional DQN trained with a KL divergence (C51, Bellemare et al. 2017).
This package implements Algorithm 1 + Table 1 literally, i.e. the categorical
formulation, and avoids class names that would over-claim either reading.

Scope
-----
Implemented: the POSG-style grid environment (parcel state, attractiveness
metric and attractiveness-weighted sampling, synchronised updates, land
preference metric, conversion-alignment metric), the UKCEH LCM data loader, the
QOLU agent, the bottom-up investor agent (BUIA) and the Section 3.5 joint loop,
the three baselines named in Section 4 (:mod:`chapter4_qolu.baselines`:
MitM, LToS and P-MADDPG, each implemented from its own source - the paper gives
one sentence each), and the four-scenario experiment runner
(:mod:`chapter4_qolu.run_scenarios`) with its reporting and figure code.

The UKCEH Land Cover Map rasters are not bundled; see ``README.md``.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
