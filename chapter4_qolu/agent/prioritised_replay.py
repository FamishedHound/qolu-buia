"""Prioritised experience replay (Schaul et al. 2016), proportional variant.

Algorithm 1 says only "sample a minibatch of transitions from D", but QOLU
combines the Rainbow components with a C51 head, and prioritised replay is one
of Rainbow's six components (Hessel et al. 2018).  It is therefore implemented
here, behind ``QOLUConfig.prioritised_replay``, rather than being read into the
pseudocode: with the switch off the agent keeps the uniform
:class:`~chapter4_qolu.agent.replay.NStepReplayBuffer` unchanged.

What is prioritised
-------------------
The *n-step* transition, not the raw environment step: this class subclasses
:class:`~chapter4_qolu.agent.replay.NStepReplayBuffer` and overrides only where
a completed transition is stored, so Algorithm 1's ``R_t`` accumulation and its
``gamma^n`` discount are shared with the uniform buffer.

The mechanics follow Schaul et al. section 3.3 and Rainbow's settings:

* ``P(i) = p_i^alpha / sum_j p_j^alpha`` with ``p_i`` = the per-sample KL loss
  plus a small ``epsilon`` so nothing is starved; a sum tree makes sampling and
  updating ``O(log N)``;
* a *stratified* draw - ``[0, p_total)`` is cut into ``batch_size`` equal
  segments and one uniform sample is taken inside each (Schaul et al.,
  "Efficient Implementation");
* importance-sampling weights ``w_i = (N P(i))^-beta`` with ``beta`` annealed
  from ``beta_start`` to ``beta_end``, normalised by ``max_i w_i`` so the
  update is only ever scaled down;
* new transitions enter at the largest priority seen so far, so every
  transition is replayed at least once.

Design note (alpha, beta): the paper fixes none of these.  ``alpha = 0.5`` and
``beta: 0.4 -> 1.0`` are Rainbow's published values for the distributional
variant; ``epsilon = 1e-6`` is the usual guard.

Design note (IS normalisation): Schaul et al. normalise by ``1 / max_i w_i``
over the whole buffer; as in every reference implementation the maximum is
taken over the *minibatch*, which is what keeps the largest weight in a batch
at exactly 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from chapter4_qolu.agent.replay import (
    NStepReplayBuffer,
    Transition,
    TransitionBatch,
    stack_transitions,
)

__all__ = [
    "BetaSchedule",
    "PrioritisedBatch",
    "PrioritisedReplayBuffer",
    "SumTree",
    "importance_sampling_weights",
]


class SumTree:
    """Fixed-capacity array of non-negative priorities with prefix-sum search."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        leaves = 1
        while leaves < self.capacity:
            leaves *= 2
        self._leaves = leaves
        #: 1-indexed heap; leaf ``i`` lives at ``_leaves + i``.
        self._nodes = np.zeros(2 * leaves, dtype=np.float64)

    def __len__(self) -> int:
        return self.capacity

    def __getitem__(self, index: int) -> float:
        return float(self._nodes[self._leaf(index)])

    def __setitem__(self, index: int, value: float) -> None:
        value = float(value)
        if value < 0.0 or not np.isfinite(value):
            raise ValueError("priorities must be finite and non-negative")
        node = self._leaf(index)
        self._nodes[node] = value
        node //= 2
        while node >= 1:
            self._nodes[node] = self._nodes[2 * node] + self._nodes[2 * node + 1]
            node //= 2

    @property
    def total(self) -> float:
        """``sum_i p_i``."""
        return float(self._nodes[1])

    @property
    def maximum(self) -> float:
        """The largest stored priority."""
        return float(self._nodes[self._leaves : self._leaves + self.capacity].max())

    def values(self) -> np.ndarray:
        """A copy of the leaves, in slot order."""
        return np.array(self._nodes[self._leaves : self._leaves + self.capacity])

    def find(self, prefix: float) -> int:
        """The slot whose cumulative-priority interval contains ``prefix``."""
        prefix = min(max(float(prefix), 0.0), self.total)
        node = 1
        while node < self._leaves:
            left = 2 * node
            left_sum = self._nodes[left]
            if prefix <= left_sum:
                node = left
            else:
                prefix -= left_sum
                node = left + 1
        return min(node - self._leaves, self.capacity - 1)

    def clear(self) -> None:
        self._nodes.fill(0.0)

    def _leaf(self, index: int) -> int:
        index = int(index)
        if not 0 <= index < self.capacity:
            raise IndexError(f"slot {index} is outside a capacity of {self.capacity}")
        return self._leaves + index


@dataclass(frozen=True)
class BetaSchedule:
    """Linear annealing of the importance-sampling exponent ``beta``."""

    start: float = 0.4
    end: float = 1.0
    steps: int = 1_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.start <= self.end <= 1.0:
            raise ValueError("require 0 <= beta_start <= beta_end <= 1")
        if self.steps < 1:
            raise ValueError("steps must be positive")

    def __call__(self, step: int) -> float:
        if step <= 0:
            return float(self.start)
        if step >= self.steps:
            return float(self.end)
        return float(self.start + (step / self.steps) * (self.end - self.start))


def importance_sampling_weights(
    probabilities: np.ndarray, buffer_size: int, beta: float
) -> np.ndarray:
    """``w_i = (N P(i))^-beta``, normalised so the largest weight is one."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.size == 0:
        raise ValueError("at least one probability is required")
    if np.any(probabilities <= 0.0):
        raise ValueError("sampling probabilities must be positive")
    if buffer_size < 1:
        raise ValueError("buffer_size must be positive")
    weights = np.power(buffer_size * probabilities, -float(beta))
    return weights / weights.max()


@dataclass(frozen=True)
class PrioritisedBatch(TransitionBatch):
    """A minibatch plus what is needed to correct and re-prioritise it."""

    indices: np.ndarray
    probabilities: np.ndarray
    weights: np.ndarray


class PrioritisedReplayBuffer(NStepReplayBuffer):
    """n-step replay that samples in proportion to each transition's KL loss."""

    def __init__(
        self,
        capacity: int,
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.5,
        epsilon: float = 1e-6,
        beta: BetaSchedule | None = None,
    ) -> None:
        super().__init__(capacity=capacity, n_step=n_step, gamma=gamma)
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.beta_schedule = beta if beta is not None else BetaSchedule()
        self.tree = SumTree(capacity)
        #: The base class's deque storage is left unused on purpose: priorities
        #: need *stable* slot indices, and a deque renumbers its contents every
        #: time it evicts, so the n-step transitions go into a fixed-size ring.
        self._items: list[Transition | None] = [None] * capacity
        self._item_discounts = np.zeros(capacity, dtype=np.float64)
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._cursor = 0
        self._size = 0
        self._batches_drawn = 0
        self.max_priority = 1.0

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._size

    def priorities(self) -> list[float]:
        """The raw priorities (KL + epsilon), in slot order, one per stored item."""
        return [float(value) for value in self._priorities[: self._size]]

    def beta_now(self) -> float:
        """``beta`` for the next minibatch, from this buffer's own schedule."""
        return float(self.beta_schedule(self._batches_drawn))

    # ------------------------------------------------------------------
    def _store(self, transition: Transition, discount: float) -> None:
        """Insert an n-step transition at the largest priority seen so far."""
        slot = self._cursor
        self._items[slot] = transition
        self._item_discounts[slot] = discount
        self._set_priority(slot, self.max_priority)
        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _set_priority(self, slot: int, priority: float) -> None:
        self._priorities[slot] = priority
        self.tree[slot] = priority**self.alpha

    def update_priorities(
        self, indices: Sequence[int], priorities: Sequence[float]
    ) -> None:
        """Re-prioritise sampled transitions by their per-sample KL loss."""
        indices = list(indices)
        values = [float(value) for value in priorities]
        if len(indices) != len(values):
            raise ValueError("one priority is required per index")
        if any(value < 0.0 or not np.isfinite(value) for value in values):
            raise ValueError("priorities must be finite and non-negative")
        for slot, value in zip(indices, values):
            priority = value + self.epsilon
            self._set_priority(int(slot), priority)
            self.max_priority = max(self.max_priority, priority)

    # ------------------------------------------------------------------
    def sample(  # type: ignore[override]
        self,
        batch_size: int,
        rng: np.random.Generator,
        beta: float | None = None,
    ) -> PrioritisedBatch:
        """Draw ``batch_size`` transitions with probability ``p^alpha / sum p^alpha``."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        beta = self.beta_now() if beta is None else float(beta)
        self._batches_drawn += 1

        total = self.tree.total
        if total <= 0.0:  # pragma: no cover - only if every priority is exactly 0
            indices = rng.integers(0, self._size, size=batch_size)
        else:
            segment = total / batch_size
            offsets = (np.arange(batch_size) + rng.random(batch_size)) * segment
            indices = np.array(
                [self.tree.find(float(offset)) for offset in offsets], dtype=np.int64
            )

        indices = np.clip(indices, 0, self._size - 1).astype(np.int64)
        if total > 0.0:
            probabilities = self.tree.values()[indices] / total
        else:  # pragma: no cover - only reachable with every priority at zero
            probabilities = np.full(batch_size, 1.0 / self._size)
        # Guard the ``(N P)^-beta`` division: a slot is only reachable when its
        # priority is positive, but floating-point underflow must not divide by 0.
        probabilities = np.maximum(probabilities, np.finfo(np.float64).tiny)

        items: list[Transition] = []
        for index in indices:
            item = self._items[int(index)]
            if item is None:  # pragma: no cover - slots below _size are filled
                raise RuntimeError(f"replay slot {int(index)} is empty")
            items.append(item)
        batch = stack_transitions(items, self._item_discounts[indices].tolist())
        return PrioritisedBatch(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            next_observations=batch.next_observations,
            dones=batch.dones,
            discounts=batch.discounts,
            indices=indices,
            probabilities=probabilities,
            weights=importance_sampling_weights(probabilities, self._size, beta).astype(
                np.float32
            ),
        )

    def clear(self) -> None:
        super().clear()
        self.tree.clear()
        self._items = [None] * self.capacity
        self._item_discounts.fill(0.0)
        self._priorities.fill(0.0)
        self._cursor = 0
        self._size = 0
        self._batches_drawn = 0
        self.max_priority = 1.0
