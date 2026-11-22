"""Replay buffer with n-step returns (Algorithm 1, lines 7 and 12).

Algorithm 1 stores transitions ``(s, a, r, s')`` in a buffer ``D`` and then
computes ``R_t = sum_{i=0}^{n-1} gamma^i r_{t+i} + gamma^n max_a' Q(...)``.
The bootstrap half of that expression is applied by the agent (with double
Q-learning and the categorical projection); this module owns the accumulation
of the first half plus the matching ``gamma^n`` discount.

Uniform sampling.  Algorithm 1 says only "sample a minibatch of transitions
from D"; no prioritisation is described anywhere in the paper.  Rainbow-style
agents (Hessel et al. 2018) do prioritise their replay, so the
prioritised reading lives beside this one in
:mod:`chapter4_qolu.agent.prioritised_replay`, which subclasses this buffer so
that the *n-step* transition stays the unit that is stored, sampled and
prioritised.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "NStepReplayBuffer",
    "Transition",
    "TransitionBatch",
    "n_step_return",
    "stack_transitions",
]


@dataclass(frozen=True)
class Transition:
    """A single environment transition ``(s, a, r, s')`` plus its done flag."""

    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


@dataclass(frozen=True)
class TransitionBatch:
    """A minibatch of n-step transitions."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    discounts: np.ndarray


def n_step_return(rewards: Sequence[float], gamma: float) -> float:
    """``sum_{i=0}^{n-1} gamma^i r_{t+i}``."""
    total = 0.0
    discount = 1.0
    for reward in rewards:
        total += discount * float(reward)
        discount *= gamma
    return total


def stack_transitions(
    items: Sequence[Transition], discounts: Sequence[float]
) -> TransitionBatch:
    """Collate stored n-step transitions into the arrays the agent consumes."""
    return TransitionBatch(
        observations=np.stack(
            [np.asarray(item.observation, dtype=np.float32) for item in items]
        ),
        actions=np.asarray([item.action for item in items], dtype=np.int64),
        rewards=np.asarray([item.reward for item in items], dtype=np.float32),
        next_observations=np.stack(
            [np.asarray(item.next_observation, dtype=np.float32) for item in items]
        ),
        dones=np.asarray([float(item.done) for item in items], dtype=np.float32),
        discounts=np.asarray(discounts, dtype=np.float32),
    )


class NStepReplayBuffer:
    """Fixed-capacity buffer that emits n-step transitions."""

    def __init__(self, capacity: int, n_step: int = 3, gamma: float = 0.99) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if n_step < 1:
            raise ValueError("n_step must be positive")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self._storage: deque[Transition] = deque(maxlen=capacity)
        self._discounts: deque[float] = deque(maxlen=capacity)
        self._window: deque[Transition] = deque()

    def __len__(self) -> int:
        return len(self._storage)

    def can_sample(self, batch_size: int) -> bool:
        return len(self) >= batch_size >= 1

    def push(self, transition: Transition) -> None:
        """Add one environment transition, emitting n-step transitions as they complete."""
        self._window.append(transition)
        if transition.done:
            while self._window:
                self._emit()
                self._window.popleft()
        elif len(self._window) == self.n_step:
            self._emit()
            self._window.popleft()

    def _emit(self) -> None:
        first = self._window[0]
        last = self._window[-1]
        rewards = [item.reward for item in self._window]
        self._store(
            Transition(
                observation=first.observation,
                action=first.action,
                reward=n_step_return(rewards, self.gamma),
                next_observation=last.next_observation,
                done=last.done,
            ),
            self.gamma ** len(self._window),
        )

    def _store(self, transition: Transition, discount: float) -> None:
        """Add one completed n-step transition to storage.

        The single seam a prioritised subclass needs: everything above it -
        the n-step window, the reward accumulation and the ``gamma^n``
        discount - is shared, and only *where the transition goes* differs.
        """
        self._storage.append(transition)
        self._discounts.append(discount)

    def sample(self, batch_size: int, rng: np.random.Generator) -> TransitionBatch:
        """Uniformly sample ``batch_size`` stored n-step transitions."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self._storage:
            raise ValueError("cannot sample from an empty replay buffer")
        indices = rng.integers(0, len(self._storage), size=batch_size)
        items = [self._storage[int(index)] for index in indices]
        discounts = [self._discounts[int(index)] for index in indices]
        return stack_transitions(items, discounts)

    def clear(self) -> None:
        self._storage.clear()
        self._discounts.clear()
        self._window.clear()
