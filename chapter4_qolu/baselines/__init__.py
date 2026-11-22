"""The three baselines named in Section 4 of the IJCAI-2025 paper.

    "We propose three baselines from the state-of-the-art for our experiments:
     (1) the Modelling-in-the-Middle (MitM)'s approach, proposed by Bone et al.
     [2011]; (2) the LToS algorithm, proposed by Yi et al. [2021]; (3) the
     P-MADDPG, proposed by Pelcner et al. [2024]."

That sentence is the *entire* specification the paper gives, so each baseline
is implemented from its own source publication.  Each module states what is
published, what is inferred and what is assumed; the research note behind the
LToS and MitM implementations is
``chapter4_qolu/docs/baselines_lto_s_and_mitm_research.md``.

Every method implements :class:`chapter4_qolu.baselines.base.BaselineMethod` and
is driven by :func:`chapter4_qolu.baselines.base.run_baseline`, so all of them
run inside the *same* :class:`~chapter4_qolu.env.urban_env.UrbanLandUseEnv` with
the same scale settings (``grid_size``, ``conversion_fraction``,
``max_conversions_per_iteration``), the same seeds and the same metrics from
:mod:`chapter4_qolu.env.evaluation`.  The QOLU + BUIA system keeps its own entry
point (:func:`chapter4_qolu.runner.train_joint`); nothing here changes it.
"""

from __future__ import annotations

from chapter4_qolu.baselines.base import (
    BaselineDecision,
    BaselineMethod,
    BaselineReport,
    conversion_budget,
    run_baseline,
)
from chapter4_qolu.baselines.config import LToSConfig, MitMConfig, PMADDPGConfig
from chapter4_qolu.baselines.ltos import FixedLToSMethod, LToSMethod, fixed_ltos
from chapter4_qolu.baselines.mitm import MitMMethod, learned_mitm
from chapter4_qolu.baselines.pmaddpg import (
    PMADDPG_VARIANTS,
    PMADDPGMethod,
    pmaddpg_reg_only,
    pmaddpg_v1,
    published_pmaddpg,
)

#: The methods a scenario runner can iterate over, in the paper's own order.
BASELINE_METHODS: tuple[type[BaselineMethod], ...] = (
    MitMMethod,
    LToSMethod,
    PMADDPGMethod,
)

__all__ = [
    "BASELINE_METHODS",
    "PMADDPG_VARIANTS",
    "BaselineDecision",
    "BaselineMethod",
    "BaselineReport",
    "FixedLToSMethod",
    "LToSConfig",
    "LToSMethod",
    "MitMConfig",
    "MitMMethod",
    "PMADDPGConfig",
    "PMADDPGMethod",
    "conversion_budget",
    "fixed_ltos",
    "learned_mitm",
    "pmaddpg_reg_only",
    "pmaddpg_v1",
    "published_pmaddpg",
    "run_baseline",
]
