"""Ornstein-Uhlenbeck exploration noise, shared by two of the baselines.

Both baselines need temporally correlated exploration noise on a
*continuous* action:

* P-MADDPG perturbs each stakeholder's per-parcel approve-score (DDPG's
  standard exploration, inherited from MADDPG);
* LToS perturbs the high-level sharing weights ``w_i^out`` - the published
  traffic-6x6 column of Yi et al. (2021) gives "Ornstein-Uhlenbeck, sigma =
  0.25".

Design note: ``theta`` and ``dt`` are not published by either source, so the
canonical DDPG values (Lillicrap et al. 2016) are used: ``theta = 0.15`` with a
time step of ``1e-2``.  Only ``sigma`` - which *is* published for LToS - changes
between the two users.
"""

from __future__ import annotations

import numpy as np

__all__ = ["OrnsteinUhlenbeckNoise"]


class OrnsteinUhlenbeckNoise:
    """Mean-reverting Gaussian process ``dx = theta (mu - x) dt + sigma dW``."""

    def __init__(
        self,
        size: int,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
        rng: np.random.Generator | None = None,
    ) -> None:
        if size < 1:
            raise ValueError("size must be positive")
        if theta < 0.0:
            raise ValueError("theta must be non-negative")
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.size = int(size)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.rng = rng or np.random.default_rng()
        self.state = np.zeros(self.size, dtype=np.float64)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Return the process to the origin, optionally re-seeding it."""
        self.state = np.zeros(self.size, dtype=np.float64)
        if rng is not None:
            self.rng = rng

    def sample(self) -> np.ndarray:
        """Advance the process by ``dt`` and return the new state.

        With ``sigma = 0`` the process never leaves the origin, which is how the
        tests and the greedy evaluation mode switch exploration off.
        """
        increment = -self.theta * self.state * self.dt + self.sigma * np.sqrt(
            self.dt
        ) * self.rng.normal(size=self.size)
        self.state = self.state + increment
        return np.array(self.state, dtype=np.float64, copy=True)
