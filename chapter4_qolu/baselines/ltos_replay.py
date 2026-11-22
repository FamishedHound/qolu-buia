"""Replay buffer for the LToS baseline: one entry per whole-lattice timestep.

Appendix C of Yi et al. (2021) lists a **unified PRNG / shared seed** so that
every agent samples the *same* timesteps from its own buffer - required because
``w_i^in`` has to be consistent across a mini-batch.  Storing one entry per
timestep, with every district's observation, sharing vector, action and shaped
reward inside it, makes that synchronisation structural rather than a
convention: a mini-batch is a set of timesteps and all agents necessarily share
it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields

import numpy as np

__all__ = ["LToSTransition", "LToSTransitionBatch", "LatticeReplayBuffer"]


@dataclass(frozen=True)
class LToSTransition:
    """One iteration of the whole district lattice."""

    node_features: np.ndarray
    sharing_in: np.ndarray
    parcel_features: np.ndarray
    parcel_districts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_node_features: np.ndarray
    next_sharing_in: np.ndarray
    next_parcel_features: np.ndarray
    next_parcel_districts: np.ndarray
    done: bool


@dataclass(frozen=True)
class LToSTransitionBatch:
    """``batch_size`` stacked timesteps; the leading axis is the mini-batch."""

    node_features: np.ndarray
    sharing_in: np.ndarray
    parcel_features: np.ndarray
    parcel_districts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_node_features: np.ndarray
    next_sharing_in: np.ndarray
    next_parcel_features: np.ndarray
    next_parcel_districts: np.ndarray
    dones: np.ndarray


_STACKED = tuple(field.name for field in fields(LToSTransition) if field.name != "done")


class LatticeReplayBuffer:
    """Fixed-capacity, uniformly sampled buffer of lattice timesteps."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._storage: deque[LToSTransition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._storage)

    def push(self, transition: LToSTransition) -> None:
        self._storage.append(transition)

    def sample(self, batch_size: int, rng: np.random.Generator) -> LToSTransitionBatch:
        """Draw ``batch_size`` timesteps; every district sees the same ones."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self._storage:
            raise ValueError("cannot sample from an empty replay buffer")
        indices = rng.integers(0, len(self._storage), size=batch_size)
        items = [self._storage[int(index)] for index in indices]
        stacked = {
            name: np.stack([getattr(item, name) for item in items]) for name in _STACKED
        }
        return LToSTransitionBatch(
            dones=np.asarray([float(item.done) for item in items], dtype=np.float32),
            **stacked,
        )

    def clear(self) -> None:
        self._storage.clear()
