"""Transition rewards for a QOLU stakeholder's appraisals.

Implements ``R_QOLU,i = f(parcel_data, objectives_i)`` of Section 3.5 as a
per-parcel, per-action signal.  The three readings of the paper's reward
allocation and the reasoning behind the default are documented on
:class:`chapter4_qolu.config.QOLURewardConfig`.

The optional ``scale_by_scenario_weight`` reading of Section 3.1's
``sum_i w_i R_phi_i`` is applied here, at the one place a stakeholder's own
reward is produced, so that :func:`chapter4_qolu.runner.train_joint` and
:class:`chapter4_qolu.baselines.pmaddpg.PMADDPGMethod` cannot drift apart.  The
scenario weighting and the stakeholder count are *passed in*: nothing in this
module reads a scenario from module state.
"""

from __future__ import annotations

from collections.abc import Container, Sequence

import numpy as np

from chapter4_qolu.config import QOLURewardConfig, RewardConfig, ScenarioWeights
from chapter4_qolu.env.compliance import compliance_scores
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.objectives import TopDownObjective

__all__ = ["qolu_parcel_rewards", "scenario_weight_multiplier"]

#: Mirrors chapter4_qolu.env.urban_env.APPROVE (imported here would be circular).
_APPROVE = 1


def scenario_weight_multiplier(
    weights: ScenarioWeights, objective: TopDownObjective, n_stakeholders: int
) -> float:
    """How much of ``sum_i w_i R_phi_i`` one stakeholder's own reward carries.

    ``n_stakeholders * w_i`` with the weighting first normalised to sum to one,
    so the multipliers average exactly 1 whatever the scenario and an equal
    weighting leaves every reward untouched.

    Design note (normalisation): the published Scenario 1 weights sum to 0.99,
    because the paper rounds 1/3 to 0.33 (Table 3).  Taking ``w_i`` literally
    would shrink every Scenario 1 reward by 1% relative to Scenarios 2-4 and
    make the four scenarios differ in reward *scale* as well as in balance, so
    :meth:`~chapter4_qolu.config.ScenarioWeights.normalised` is applied first.
    """
    if n_stakeholders < 1:
        raise ValueError("n_stakeholders must be positive")
    return float(n_stakeholders) * float(weights.normalised()[objective])


def qolu_parcel_rewards(
    grid_before: ParcelGrid,
    parcels: Sequence[ParcelIndex],
    actions: np.ndarray,
    objective: TopDownObjective,
    proposed: Container[ParcelIndex] | None,
    config: RewardConfig,
    qolu_reward: QOLURewardConfig,
    team_reward: float,
    scenario_weights: ScenarioWeights | None = None,
    n_stakeholders: int | None = None,
) -> np.ndarray:
    """One reward per appraised candidate parcel.

    Args:
        grid_before: the landscape the appraisal was made in.
        parcels: the decision buffer, in the order the agent saw it.
        actions: the stakeholder's own ``REJECT``/``APPROVE`` decisions.
        objective: the objective this stakeholder represents.
        proposed: the parcels the investor put forward (``None`` = all of them).
        config: reward parameters (protected groups, thresholds, distances).
        qolu_reward: which reading of the paper's reward allocation to use.
        team_reward: the shared team reward, mixed in when configured.
        scenario_weights: the run's preference weighting ``w``.  Required only
            by ``qolu_reward.scale_by_scenario_weight``; ignored otherwise.
        n_stakeholders: how many top-down agents share the weighting.  Required
            only by ``qolu_reward.scale_by_scenario_weight``.
    """
    parcels = tuple(parcels)
    taken = np.asarray(actions, dtype=np.int64).reshape(-1)
    if taken.shape[0] != len(parcels):
        raise ValueError(
            f"got {taken.shape[0]} actions for {len(parcels)} candidate parcels"
        )
    if qolu_reward.scale_by_scenario_weight and (
        scenario_weights is None or n_stakeholders is None
    ):
        raise ValueError(
            "scale_by_scenario_weight needs both scenario_weights and "
            "n_stakeholders; pass the environment's weighting and the number "
            "of top-down agents"
        )
    if not parcels:
        return np.zeros(0, dtype=np.float64)

    compliance = compliance_scores(grid_before, parcels, objective, config)
    centred = compliance - qolu_reward.neutral_compliance

    if qolu_reward.approval_form == "symmetric":
        rewards = np.where(taken == _APPROVE, centred, -centred)
    else:  # "asymmetric": rejection is neutral
        rewards = np.where(taken == _APPROVE, centred, 0.0)

    if proposed is not None and qolu_reward.unproposed_reward_scale != 1.0:
        in_play = np.asarray([parcel in proposed for parcel in parcels], dtype=bool)
        rewards = np.where(
            in_play, rewards, rewards * qolu_reward.unproposed_reward_scale
        )

    if qolu_reward.scale_by_scenario_weight:
        # Design note: the per-parcel reading of ``R = sum_i w_i R_phi_i +
        # R_society``.  Only the stakeholder's *own* term is weighted; the
        # shared term added below already carries every w_i inside it.
        assert scenario_weights is not None and n_stakeholders is not None
        rewards = rewards * scenario_weight_multiplier(
            scenario_weights, objective, n_stakeholders
        )

    if qolu_reward.team_reward_weight > 0.0:
        rewards = rewards + qolu_reward.team_reward_weight * float(team_reward)

    return np.asarray(rewards, dtype=np.float64)
