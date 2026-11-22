"""No regulator: BUIA proposes and every proposal is converted.

The ablation removes the top-down layer and nothing else.  The investor is the
:class:`~chapter4_qolu.agent.buia_agent.BUIAAgent` QOLU + BUIA uses - built from
the same :class:`~chapter4_qolu.config.BUIAConfig` with the same seed, so its
initial weights and its sampling stream are the ones the joint run starts
from - and it sees the same attractiveness-weighted decision buffer, proposes
up to the same conversion budget and learns from the same reward,
``R_NN = Attract(s') - Attract(s)``, by the same REINFORCE step.  No
stakeholder appraises anything: every proposal is approved, so the parcels the
investor puts forward are the parcels that convert.

Design note (what is approved): exactly the proposed parcels, so the record's
``n_approved`` equals ``n_proposals`` and reads "every proposal approved".
Approving every candidate instead would convert the same parcels - only a
proposal can convert - but would report approvals of parcels nobody proposed.

Design note (order of play): :func:`~chapter4_qolu.baselines.base.run_baseline`
calls :meth:`decide`, steps the environment, then :meth:`observe` and
:meth:`train_step`; that is the investor's order in
:func:`chapter4_qolu.runner.train_joint` too - propose, convert, update on the
change in attractiveness - so the investor half of the loop is unchanged.
"""

from __future__ import annotations

import numpy as np

from chapter4_qolu.agent.buia_agent import BUIAAgent, buia_proposal_limit
from chapter4_qolu.baselines.base import BaselineDecision, BaselineMethod
from chapter4_qolu.config import BUIAConfig
from chapter4_qolu.env.urban_env import (
    APPROVE,
    REJECT,
    DecisionBatch,
    StepResult,
    UrbanLandUseEnv,
)
from chapter4_qolu.methods import NO_REGULATOR_METHOD

__all__ = ["NoRegulatorMethod"]


class NoRegulatorMethod(BaselineMethod):
    """The bottom-up investor alone, with no top-down appraisal."""

    def __init__(self, investor: BUIAAgent) -> None:
        self.investor = investor
        self._env: UrbanLandUseEnv | None = None
        self._pending_reward: float | None = None

    @classmethod
    def for_env(
        cls, env: UrbanLandUseEnv, seed: int | None = None
    ) -> NoRegulatorMethod:
        """The investor QOLU + BUIA would build for ``env`` with ``seed``."""
        config = BUIAConfig(
            observation_channels=env.buia_observation_channels,
            observation_window=env.config.buia_observation_window,
            max_proposals=buia_proposal_limit(env.config),
        )
        return cls(BUIAAgent(config, seed=seed))

    @property
    def name(self) -> str:
        return NO_REGULATOR_METHOD

    def reset(self, env: UrbanLandUseEnv, seed: int | None = None) -> None:
        """Bind to ``env``; the investor's learned parameters are kept."""
        self._env = env
        self._pending_reward = None

    def _bound_env(self) -> UrbanLandUseEnv:
        if self._env is None:
            raise RuntimeError("call reset() before decide()")
        return self._env

    def decide(self, batch: DecisionBatch) -> BaselineDecision:
        env = self._bound_env()
        candidates = tuple(dict.fromkeys(batch.parcels))
        proposal = self.investor.propose(
            candidates,
            env.buia_observations(candidates),
            max_proposals=env.config.max_conversions_per_iteration,
        )
        proposed = set(proposal.parcels)
        decisions = np.array(
            [APPROVE if parcel in proposed else REJECT for parcel in batch.parcels],
            dtype=np.int64,
        )
        return BaselineDecision(decisions=decisions, proposals=proposal.parcels)

    def observe(self, result: StepResult) -> None:
        env = self._bound_env()
        self._pending_reward = env.attractiveness_delta(
            result.conversions, mode=self.investor.config.reward_mode
        )

    def train_step(self) -> float | None:
        """One REINFORCE step on the proposal just converted."""
        if self._pending_reward is None:
            return None
        reward, self._pending_reward = self._pending_reward, None
        return float(self.investor.update(reward))
