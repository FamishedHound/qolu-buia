"""The per-parcel reward has to be learnable by the published agent.

These are behavioural tests rather than unit tests: they check that a QOLU agent
fed ``R_QOLU,i = f(parcel_data, objectives_i)`` actually learns to approve
compliant parcels and reject non-compliant ones.  The convolution stack is
shrunk so the test runs in seconds; the atom support, the loss, the dueling
head and the optimiser are the published ones, because those are what decide
whether a +-0.5 reward is learnable on a [-200, 200] support at all.
"""

from __future__ import annotations

import numpy as np

from chapter4_qolu.agent.qolu_agent import QOLUAgent, QOLUConfig
from chapter4_qolu.env.urban_env import APPROVE, REJECT

_WINDOW = 5
_CHANNELS = 2


def compliant_observation(compliance: float) -> np.ndarray:
    """Channel 0 carries the parcel's compliance, channel 1 its complement."""
    observation = np.zeros((_CHANNELS, _WINDOW, _WINDOW), dtype=np.float32)
    observation[0] = compliance
    observation[1] = 1.0 - compliance
    return observation


def parcel_reward(compliance: float, action: int) -> float:
    """The shipped symmetric form: +-(compliance - 0.5)."""
    centred = compliance - 0.5
    return centred if action == APPROVE else -centred


def make_agent() -> QOLUAgent:
    return QOLUAgent(
        QOLUConfig(
            observation_channels=_CHANNELS,
            observation_window=_WINDOW,
            conv_channels=(8, 16),
            hidden_units=64,
            batch_size=64,
            learning_starts=64,
            replay_capacity=2_000,
            target_sync_interval=50,
            n_step=1,
        ),
        seed=0,
    )


def train_on_synthetic_parcels(agent: QOLUAgent, updates: int) -> list[float]:
    """Feed terminal (parcel, action) transitions drawn from both classes."""
    rng = np.random.default_rng(0)
    losses: list[float] = []
    for _ in range(updates):
        compliance = float(rng.choice([0.0, 1.0]))
        action = int(rng.integers(0, 2))
        observation = compliant_observation(compliance)
        agent.remember(
            observation,
            action,
            parcel_reward(compliance, action),
            observation,
            True,  # one-step appraisal: the target is the immediate reward
        )
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)
    return losses


class TestPerParcelRewardIsLearnable:
    def test_agent_learns_to_approve_compliant_and_reject_the_rest(self) -> None:
        agent = make_agent()
        train_on_synthetic_parcels(agent, updates=400)

        compliant = compliant_observation(1.0)[np.newaxis]
        non_compliant = compliant_observation(0.0)[np.newaxis]
        assert int(agent.act(compliant, epsilon=0.0)[0]) == APPROVE
        assert int(agent.act(non_compliant, epsilon=0.0)[0]) == REJECT

    def test_quality_scores_order_the_two_parcel_classes(self) -> None:
        agent = make_agent()
        train_on_synthetic_parcels(agent, updates=400)
        observations = np.stack(
            [compliant_observation(1.0), compliant_observation(0.0)]
        )
        scores = agent.quality_scores(observations)
        assert scores[0] > 0.0 > scores[1]

    def test_learning_reduces_the_distributional_loss(self) -> None:
        agent = make_agent()
        losses = train_on_synthetic_parcels(agent, updates=400)
        assert len(losses) > 50
        assert float(np.mean(losses[-25:])) < float(np.mean(losses[:25]))

    def test_an_untrained_agent_does_not_already_separate_the_classes(self) -> None:
        # Guards against the test passing for reasons other than learning.
        agent = make_agent()
        observations = np.stack(
            [compliant_observation(1.0), compliant_observation(0.0)]
        )
        scores = agent.quality_scores(observations)
        assert not (scores[0] > 0.0 > scores[1])

    def test_team_style_constant_reward_teaches_nothing_about_parcels(self) -> None:
        # Pinned down: a reward that ignores the parcel and the action cannot
        # separate compliant from non-compliant parcels.
        agent = make_agent()
        rng = np.random.default_rng(0)
        for _ in range(400):
            compliance = float(rng.choice([0.0, 1.0]))
            observation = compliant_observation(compliance)
            agent.remember(observation, int(rng.integers(0, 2)), 1.0, observation, True)
            agent.learn()
        observations = np.stack(
            [compliant_observation(1.0), compliant_observation(0.0)]
        )
        scores = agent.quality_scores(observations)
        assert abs(float(scores[0] - scores[1])) < 1.0
