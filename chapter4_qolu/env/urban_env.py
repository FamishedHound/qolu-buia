"""POSG-style urban land-use environment (Section 3.5 of the paper).

One *iteration* of the simulation:

1. a decision buffer of ``samples_per_iteration`` parcels is drawn **with
   replacement**, with probability proportional to attractiveness (Eq. 2);
2. the bottom-up investor (BUIA) "proposes changes to the land parcels based on
   their objectives" - a subset of the sampled candidates, passed to
   :meth:`UrbanLandUseEnv.step` as ``proposals``;
3. the top-down (QOLU) stakeholders "appraise the solution", approving or
   rejecting each candidate;
4. the approved changes "are applied only after all decisions have been made,
   ensuring a synchronized update across the entire environment";
5. attractiveness is refreshed around the converted parcels and the shared team
   reward ``sum_i w_i R_phi_i + R_society`` is handed back to every agent.

Design note: the paper's QOLU "produces a unified quality metric indicating how
favorable the conversion of a non-residential parcel is", so its action set is
the binary appraisal {REJECT, APPROVE} of a proposed conversion.

Design note: ``proposals=None`` (no investor attached) means every sampled
candidate counts as proposed, which is the QOLU-only setting.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from chapter4_qolu.config import SCENARIOS, EnvConfig, ScenarioWeights
from chapter4_qolu.env.attractiveness import AttractivenessField
from chapter4_qolu.env.metrics import conversion_alignment
from chapter4_qolu.env.parcel_grid import Conversion, ParcelGrid, ParcelIndex
from chapter4_qolu.env.rewards import (
    compose_team_reward,
    objective_rewards,
    societal_reward,
)
from chapter4_qolu.env.sampling import sample_decision_buffer
from chapter4_qolu.env.well_being import population_at, total_capacity, well_being
from chapter4_qolu.land_cover import LandCoverGroup
from chapter4_qolu.objectives import TopDownObjective

__all__ = ["APPROVE", "REJECT", "DecisionBatch", "StepResult", "UrbanLandUseEnv"]

#: QOLU action set: appraise a proposed conversion.
REJECT: int = 0
APPROVE: int = 1


@dataclass(frozen=True)
class DecisionBatch:
    """The parcels an agent must appraise this iteration, plus observations."""

    parcels: tuple[ParcelIndex, ...]
    observations: np.ndarray


@dataclass(frozen=True)
class StepResult:
    """Outcome of one synchronised environment update."""

    parcels: tuple[ParcelIndex, ...]
    next_observations: np.ndarray
    team_reward: float
    objective_rewards: Mapping[TopDownObjective, float]
    societal_reward: float
    conversions: tuple[Conversion, ...]
    done: bool
    next_batch: DecisionBatch
    info: Mapping[str, float] = field(default_factory=dict)


class UrbanLandUseEnv:
    """Grid of 1 km^2 parcels appraised by top-down QOLU stakeholders."""

    def __init__(
        self,
        grid: ParcelGrid,
        config: EnvConfig | None = None,
        weights: ScenarioWeights | None = None,
    ) -> None:
        self._config = config or EnvConfig()
        self._weights = weights or SCENARIOS[1]
        if grid.shape != self._config.grid_shape:
            raise ValueError(
                f"grid {grid.shape} does not match config.grid_size / grid_width "
                f"{self._config.grid_shape}"
            )
        self._initial_grid = grid
        self._grid = grid
        self._attractiveness = AttractivenessField.from_grid(
            grid, self._config.attractiveness
        )
        self._previous_attractiveness = self._attractiveness
        self._rng = np.random.default_rng()
        self._iteration = 0
        self._candidates: tuple[ParcelIndex, ...] | None = None
        self._observations: np.ndarray | None = None
        # "historical land-use changes" plane seen by BUIA: 1 where a parcel has
        # already been converted during this episode, 0 elsewhere.
        self._history = np.zeros(grid.shape, dtype=np.float32)
        self._initial_capacity = total_capacity(grid, self._config.well_being)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> EnvConfig:
        return self._config

    @property
    def weights(self) -> ScenarioWeights:
        return self._weights

    @property
    def grid(self) -> ParcelGrid:
        return self._grid

    @property
    def initial_grid(self) -> ParcelGrid:
        return self._initial_grid

    @property
    def attractiveness(self) -> AttractivenessField:
        return self._attractiveness

    @property
    def previous_attractiveness(self) -> AttractivenessField:
        """The attractiveness buffer as it was before the last :meth:`step`."""
        return self._previous_attractiveness

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def observation_channels(self) -> int:
        """Land cover class planes plus one attractiveness plane."""
        return self._initial_grid.n_classes + 1

    @property
    def buia_observation_channels(self) -> int:
        """QOLU's channels plus the land-use history plane BUIA also sees."""
        return self.observation_channels + 1

    @property
    def population(self) -> float:
        """Households seeking housing at the current iteration."""
        return population_at(
            self._iteration, self._initial_capacity, self._config.well_being
        )

    @property
    def well_being(self) -> float:
        """``WB = H_occupied / H_total`` for the current landscape."""
        return well_being(
            self._grid,
            self._attractiveness,
            self.population,
            self._config.well_being,
        )

    # ------------------------------------------------------------------
    # episode control
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> DecisionBatch:
        """Restore the initial landscape and draw the first decision buffer."""
        self._rng = np.random.default_rng(seed)
        self._grid = self._initial_grid
        self._attractiveness = AttractivenessField.from_grid(
            self._initial_grid, self._config.attractiveness
        )
        self._previous_attractiveness = self._attractiveness
        self._iteration = 0
        self._history = np.zeros(self._initial_grid.shape, dtype=np.float32)
        return self._draw_candidates()

    def set_candidates(self, parcels: Iterable[ParcelIndex]) -> DecisionBatch:
        """Override this iteration's candidates (a hook for an external proposer)."""
        candidates = tuple(parcels)
        if not candidates:
            raise ValueError("at least one candidate parcel is required")
        for parcel in candidates:
            if not self._grid.contains(parcel):
                raise IndexError(
                    f"parcel {parcel} is outside the grid {self._grid.shape}"
                )
        self._candidates = candidates
        self._observations = self._observe(candidates)
        return DecisionBatch(parcels=candidates, observations=self._observations)

    def step(
        self,
        decisions: Sequence[int] | np.ndarray,
        proposals: Iterable[ParcelIndex] | None = None,
    ) -> StepResult:
        """Apply one synchronised update from the appraisals of the candidates.

        Args:
            decisions: one appraisal per candidate in the current decision
                buffer, ``REJECT`` or ``APPROVE``.
            proposals: the parcels a bottom-up investor put forward this
                iteration; only these can be converted.  ``None`` means "every
                candidate was proposed" (QOLU-only operation).
        """
        if self._candidates is None or self._observations is None:
            raise RuntimeError("call reset() before step()")
        actions = np.asarray(decisions).reshape(-1)
        if actions.size != len(self._candidates):
            raise ValueError(
                f"expected {len(self._candidates)} decisions, got {actions.size}"
            )
        proposed = self._validated_proposals(proposals)

        # Book-keeping for approval rates: how many distinct candidates this
        # appraisal approved, before the proposal filter and the cap apply.
        approved_candidates = {
            parcel
            for parcel, action in zip(self._candidates, actions)
            if int(action) == APPROVE
        }
        approved = self._approved_parcels(actions, proposed)
        grid_before = self._grid
        grid_after, conversions = grid_before.convert_to_urban(
            approved,
            fraction=self._config.conversion_fraction,
            urban_saturation=self._config.urban_saturation,
        )

        self._grid = grid_after
        self._previous_attractiveness = self._attractiveness
        if conversions:
            self._attractiveness = self._attractiveness.updated(
                grid_after, [conversion.parcel for conversion in conversions]
            )
            for conversion in conversions:
                self._history[conversion.parcel.row, conversion.parcel.col] = 1.0

        per_objective = objective_rewards(
            grid_before, grid_after, conversions, self._config.reward
        )
        societal = societal_reward(self._initial_grid, grid_after, self._config.reward)
        team_reward = compose_team_reward(per_objective, self._weights, societal)

        parcels = self._candidates
        next_observations = self._observe(parcels)
        self._iteration += 1
        done = self._iteration >= self._config.iterations

        info = {
            "conversion_alignment": conversion_alignment(grid_before, grid_after),
            "n_conversions": float(len(conversions)),
            "n_candidates": float(len(set(self._candidates))),
            "n_approved": float(len(approved_candidates)),
            "n_proposals": float(
                len(set(self._candidates)) if proposed is None else len(proposed)
            ),
            "urban_fraction": float(
                grid_after.group_fraction(LandCoverGroup.URBAN).mean()
            ),
            "mean_attractiveness": float(self._attractiveness.scores.mean()),
            "attractiveness_delta": float(
                self._attractiveness.scores.sum()
                - self._previous_attractiveness.scores.sum()
            ),
            "well_being": self.well_being,
            "population": self.population,
        }

        next_batch = self._draw_candidates()
        return StepResult(
            parcels=parcels,
            next_observations=next_observations,
            team_reward=float(team_reward),
            objective_rewards=per_objective,
            societal_reward=float(societal),
            conversions=conversions,
            done=done,
            next_batch=next_batch,
            info=info,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _validated_proposals(
        self, proposals: Iterable[ParcelIndex] | None
    ) -> frozenset[ParcelIndex] | None:
        """Check that every proposed parcel is in the current decision buffer."""
        if proposals is None:
            return None
        assert self._candidates is not None
        proposed = frozenset(proposals)
        candidates = set(self._candidates)
        unknown = proposed - candidates
        if unknown:
            raise ValueError(
                f"proposed parcels are not in this iteration's decision buffer: "
                f"{sorted(unknown)}"
            )
        return proposed

    def _approved_parcels(
        self, actions: np.ndarray, proposed: frozenset[ParcelIndex] | None
    ) -> tuple[ParcelIndex, ...]:
        """Unique approved parcels, capped at the per-iteration conversion limit.

        ``EnvConfig.max_conversions_per_iteration = None`` removes the cap, so
        every approved candidate converts.
        """
        assert self._candidates is not None
        cap = self._config.max_conversions_per_iteration
        approved: list[ParcelIndex] = []
        seen: set[ParcelIndex] = set()
        for parcel, action in zip(self._candidates, actions):
            if cap is not None and len(approved) >= cap:
                break
            if int(action) != APPROVE or parcel in seen:
                continue
            if proposed is not None and parcel not in proposed:
                continue
            seen.add(parcel)
            approved.append(parcel)
        return tuple(approved)

    def _draw_candidates(self) -> DecisionBatch:
        eligible = self._grid.convertible_mask(self._config.urban_saturation)
        if not eligible.any():
            # Fully urbanised (or empty) landscape: keep the interface stable by
            # re-proposing already-developed parcels, which convert to nothing.
            eligible = self._grid.land_mask()
        if not eligible.any():
            raise RuntimeError("landscape contains no land parcels to sample")
        candidates = sample_decision_buffer(
            self._attractiveness.scores,
            n_samples=self._config.samples_per_iteration,
            rng=self._rng,
            eligible=eligible,
        )
        self._candidates = candidates
        self._observations = self._observe(candidates)
        return DecisionBatch(parcels=candidates, observations=self._observations)

    def _observe(self, parcels: Sequence[ParcelIndex]) -> np.ndarray:
        """Stack land cover and attractiveness patches for every parcel."""
        window = self._config.observation_window
        observations = np.zeros(
            (len(parcels), self.observation_channels, window, window), dtype=np.float32
        )
        for index, parcel in enumerate(parcels):
            observations[index, :-1] = self._grid.patch(parcel, window)
            observations[index, -1] = self._attractiveness.patch(parcel, window)
        return observations

    def buia_observations(self, parcels: Sequence[ParcelIndex]) -> np.ndarray:
        """Local context ``X`` for the investor: cover, attractiveness, history.

        The window is BUIA's 2 km radius rather than the top-down window, and a
        third block of channels carries the "historical land-use changes" the
        paper says BUIA exploits - here, which parcels have already been
        converted during this episode.
        """
        window = self._config.buia_observation_window
        n_classes = self._grid.n_classes
        observations = np.zeros(
            (len(parcels), self.buia_observation_channels, window, window),
            dtype=np.float32,
        )
        for index, parcel in enumerate(parcels):
            observations[index, :n_classes] = self._grid.patch(parcel, window)
            observations[index, n_classes] = self._attractiveness.patch(parcel, window)
            observations[index, n_classes + 1] = _patch_of(
                self._history, parcel, window
            )
        return observations

    def attractiveness_delta(
        self, conversions: Sequence[Conversion], mode: str = "total"
    ) -> float:
        """``R_NN = Attract(s') - Attract(s)`` for the last synchronised update.

        Design note: "total" reads ``Attract(s)`` as the sum over the whole
        attractiveness buffer ``B_attractiveness``; "local" averages the change
        over the neighbourhoods of the parcels that were actually converted.
        """
        before = self._previous_attractiveness.scores
        after = self._attractiveness.scores
        if mode == "total":
            return float(after.sum() - before.sum())
        if mode != "local":
            raise ValueError("mode must be 'total' or 'local'")
        if not conversions:
            return 0.0
        radius = self._config.attractiveness.radius
        changed = np.zeros(self._grid.shape, dtype=bool)
        height, width = self._grid.shape
        for conversion in conversions:
            row_slice = slice(
                max(conversion.parcel.row - radius, 0),
                min(conversion.parcel.row + radius + 1, height),
            )
            col_slice = slice(
                max(conversion.parcel.col - radius, 0),
                min(conversion.parcel.col + radius + 1, width),
            )
            changed[row_slice, col_slice] = True
        return float((after - before)[changed].mean())

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_composition(
        cls,
        composition: np.ndarray,
        config: EnvConfig | None = None,
        weights: ScenarioWeights | None = None,
    ) -> UrbanLandUseEnv:
        return cls(
            ParcelGrid.from_fractions(composition), config=config, weights=weights
        )


def _patch_of(plane: np.ndarray, parcel: ParcelIndex, window: int) -> np.ndarray:
    """Zero-padded ``(window, window)`` patch of a single 2-D plane."""
    radius = window // 2
    height, width = plane.shape
    patch = np.zeros((window, window), dtype=np.float32)
    row_start = max(parcel.row - radius, 0)
    row_end = min(parcel.row + radius + 1, height)
    col_start = max(parcel.col - radius, 0)
    col_end = min(parcel.col + radius + 1, width)
    patch[
        row_start - (parcel.row - radius) : row_end - (parcel.row - radius),
        col_start - (parcel.col - radius) : col_end - (parcel.col - radius),
    ] = plane[row_start:row_end, col_start:col_end]
    return patch
