"""Networks and replay buffer of the P-MADDPG baseline (Chapter 3).

Chapter 3's defining move is the *spatial decomposition*: "this calculation
operates over per-cell state and action representations rather than agent-level
vectors".  Everything here is therefore shaped per **cell** - one row of a
mini-batch is one candidate parcel, carrying that parcel's observation, the
joint action the stakeholders took on it, and the reward each of them earned
for it.

* :class:`ParcelActor` is ``upsilon_i``: observation patch -> one approve-score
  in [0, 1] for that parcel.
* :class:`CentralisedCritic` is ``psi_i``: the per-cell state plus *all*
  stakeholders' actions on that cell -> one Q-value.  This is MADDPG's
  centralised critic, restricted to the cell the action applies to.
* :class:`JointReplayBuffer` is ``Lambda``, sampled uniformly for mini-batches
  ``{s, a, s', r}``.

Design note: the three stakeholders appraise the same parcel through the same
observation window (the published QOLU window), so the "joint per-cell state" is
that single patch rather than a concatenation of three identical copies.  The
joint *action* is genuinely an ``n_agents``-vector.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

__all__ = [
    "CentralisedCritic",
    "JointReplayBuffer",
    "JointTransition",
    "JointTransitionBatch",
    "ParcelActor",
]


def _conv_encoder(
    in_channels: int,
    window: int,
    conv_channels: tuple[int, ...],
    kernel_size: int,
    padding: int,
) -> tuple[nn.Sequential, int]:
    """Convolutional trunk plus the flattened feature count it produces."""
    layers: list[nn.Module] = []
    channels = in_channels
    for out_channels in conv_channels:
        layers.append(
            nn.Conv2d(channels, out_channels, kernel_size=kernel_size, padding=padding)
        )
        layers.append(nn.ReLU(inplace=True))
        channels = out_channels
    layers.append(nn.Flatten())
    encoder = nn.Sequential(*layers)
    with torch.no_grad():
        features = int(encoder(torch.zeros(1, in_channels, window, window)).shape[1])
    return encoder, features


class ParcelActor(nn.Module):
    """``upsilon_i``: a continuous approve-score in [0, 1] for one parcel.

    The squashing sigmoid is applied in :meth:`forward` rather than sitting in
    ``head`` as an :class:`torch.nn.Sigmoid` layer, so that :meth:`logits` can
    hand out the *pre-activation* value.  The actor loss needs it: Lowe et al.'s
    MADDPG reference implementation penalises the squared pre-activation, which
    is what keeps the sigmoid off its flat tails (see
    :meth:`~chapter4_qolu.baselines.pmaddpg_agent.PMADDPGAgent.update_actor`).
    ``head.0`` and ``head.2`` are the two linear layers; ``nn.Sigmoid`` holds no
    parameters and draws no random numbers, so applying it in :meth:`forward`
    rather than inside ``head`` leaves the parameters and their initialisation
    as they would be with a sigmoid layer.
    """

    def __init__(
        self,
        in_channels: int,
        window: int,
        conv_channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        padding: int = 1,
        hidden_units: int = 256,
    ) -> None:
        super().__init__()
        self.encoder, features = _conv_encoder(
            in_channels, window, conv_channels, kernel_size, padding
        )
        self.head = nn.Sequential(
            nn.Linear(features, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, 1),
        )

    def logits(self, observations: torch.Tensor) -> torch.Tensor:
        """The un-squashed score, shape ``(batch, 1)``."""
        if observations.ndim != 4:
            raise ValueError(
                "observations must have shape (batch, channels, window, window)"
            )
        return self.head(self.encoder(observations))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(observations))


class CentralisedCritic(nn.Module):
    """``psi_i``: per-cell state plus every stakeholder's action on that cell."""

    def __init__(
        self,
        in_channels: int,
        window: int,
        n_agents: int,
        conv_channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        padding: int = 1,
        hidden_units: int = 256,
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.encoder, features = _conv_encoder(
            in_channels, window, conv_channels, kernel_size, padding
        )
        self.head = nn.Sequential(
            nn.Linear(features + self.n_agents, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, 1),
        )

    def forward(
        self, observations: torch.Tensor, joint_actions: torch.Tensor
    ) -> torch.Tensor:
        if observations.ndim != 4:
            raise ValueError(
                "observations must have shape (batch, channels, window, window)"
            )
        if joint_actions.ndim != 2 or joint_actions.shape[1] != self.n_agents:
            raise ValueError(
                f"joint actions must have shape (batch, {self.n_agents}), "
                f"got {tuple(joint_actions.shape)}"
            )
        features = self.encoder(observations)
        return self.head(torch.cat([features, joint_actions], dim=1))


@dataclass(frozen=True)
class JointTransition:
    """One cell's experience ``{s, a, s', r}`` for all stakeholders at once."""

    observation: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observation: np.ndarray
    done: bool


@dataclass(frozen=True)
class JointTransitionBatch:
    """A uniformly sampled mini-batch ``delta ~ Lambda``."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class JointReplayBuffer:
    """``Lambda``: fixed-capacity buffer sampled uniformly (Chapter 3, line 2)."""

    def __init__(self, capacity: int, n_agents: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if n_agents < 1:
            raise ValueError("n_agents must be positive")
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self._storage: deque[JointTransition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._storage)

    def push(self, transition: JointTransition) -> None:
        actions = np.asarray(transition.actions, dtype=np.float32).reshape(-1)
        rewards = np.asarray(transition.rewards, dtype=np.float32).reshape(-1)
        if actions.size != self.n_agents or rewards.size != self.n_agents:
            raise ValueError(
                f"expected {self.n_agents} actions and rewards, got "
                f"{actions.size} and {rewards.size}"
            )
        self._storage.append(
            JointTransition(
                observation=np.asarray(transition.observation, dtype=np.float32),
                actions=actions,
                rewards=rewards,
                next_observation=np.asarray(
                    transition.next_observation, dtype=np.float32
                ),
                done=bool(transition.done),
            )
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> JointTransitionBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self._storage:
            raise ValueError("cannot sample from an empty replay buffer")
        indices = rng.integers(0, len(self._storage), size=batch_size)
        items = [self._storage[int(index)] for index in indices]
        return JointTransitionBatch(
            observations=np.stack([item.observation for item in items]),
            actions=np.stack([item.actions for item in items]),
            rewards=np.stack([item.rewards for item in items]),
            next_observations=np.stack([item.next_observation for item in items]),
            dones=np.asarray([float(item.done) for item in items], dtype=np.float32),
        )

    def clear(self) -> None:
        self._storage.clear()
