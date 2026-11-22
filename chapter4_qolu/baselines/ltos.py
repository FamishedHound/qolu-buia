"""LToS baseline - Learning to Share in Multi-Agent RL (Yi et al., 2021).

Mapping (A) of ``docs/baselines_lto_s_and_mitm_research.md``: the parcel grid is
tiled into a **district lattice** (50 x 50 -> 10 x 10 districts of 5 x 5
parcels, configurable) with rook adjacency plus a self-loop, and one LToS agent
sits on each district.

* **Low level** ``pi_i : (o_i, w_i^in) -> a_i`` - a DGN Q-network (observation
  encoder MLP -> multi-head graph attention over ``N_i`` -> Q head) over the
  discrete decision convert / skip for each candidate parcel inside district
  ``i``.
* **High level** ``phi_i : o_i -> w_i^out`` - a DDPG actor with a softmax over
  ``|N_i|``, initialised by Appendix C's "selfishness" initialiser and explored
  with Ornstein-Uhlenbeck noise.
* **Shaped reward** ``r_i^w = sum_{j in N_i} w_ji r_j`` over the district
  decomposition of the published team reward
  (:mod:`chapter4_qolu.baselines.ltos_rewards`), whose sum is provably the
  global team reward.
* **Bi-level update** following the note's Algorithm 1 sketch, first order: the
  DGN Q-network doubles as the DDPG critic and the high-level gradient is taken
  w.r.t. ``w_i^in`` at the argmax low-level action, then exchanged into
  ``g_i^out`` before the actor step.

LToS is **single-objective**: it only ever sees the scalarised team reward for
the scenario being run, so - exactly as in the original paper - it has to be
retrained once per scenario.  ``P_society``: the land-preference term of Eq. 3
is not part of the environment's reward and therefore not in the decomposition;
the environment's ``R_society`` reward term *is*, split equally across districts
so the decomposition stays conservative (see ``ltos_rewards``).

Design note (parameter sharing): DGN is by construction *one* graph-convolutional
network evaluated at every node, so the low level - and, for symmetry, the
high-level actor - use a single shared parameter set rather than 100 private
copies.  Yi et al. never state per-agent parameter copies, and 100 copies would
be unlearnable at this sample budget.

Design note (conversion budget): the per-iteration budget is a *global*
constraint LToS has no mechanism for.  Option (i) of the note is taken: the
districts decide freely and the top-B convert decisions, ranked by the
Q-advantage ``Q(convert) - Q(skip)``, are the ones executed.  The action stored
in the replay buffer is the **executed** one, so the Q-function learns the
dynamics the environment actually applies.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from chapter4_qolu.baselines.base import (
    BaselineDecision,
    BaselineMethod,
    conversion_budget,
    top_k_by_score,
)
from chapter4_qolu.baselines.config import LToSConfig
from chapter4_qolu.baselines.districts import (
    N_SLOTS,
    NODE_FEATURE_DIM,
    PARCEL_FEATURE_DIM,
    DistrictLattice,
    incoming_weights,
    node_features,
    parcel_features,
)
from chapter4_qolu.baselines.ltos_learning import (
    GraphTensors,
    soft_update,
    update_high_level,
    update_low_level,
)
from chapter4_qolu.baselines.ltos_networks import DGNQNetwork, SharingActor
from chapter4_qolu.baselines.ltos_replay import LatticeReplayBuffer, LToSTransition
from chapter4_qolu.baselines.ltos_rewards import district_rewards, shaped_rewards
from chapter4_qolu.baselines.noise import OrnsteinUhlenbeckNoise
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.env.urban_env import (
    APPROVE,
    REJECT,
    DecisionBatch,
    StepResult,
    UrbanLandUseEnv,
)

__all__ = ["FixedLToSMethod", "LToSMethod", "fixed_ltos"]

_CONVERT = 1
_SKIP = 0


class LToSMethod(BaselineMethod):
    """LToS over a district lattice, as a pluggable :class:`BaselineMethod`."""

    def __init__(self, config: LToSConfig, seed: int | None = None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        def build_q() -> DGNQNetwork:
            return DGNQNetwork(
                node_dim=NODE_FEATURE_DIM,
                parcel_dim=PARCEL_FEATURE_DIM,
                hidden_units=config.hidden_units,
                heads=config.attention_heads,
                attention_layers=config.attention_layers,
                encoder_layers=config.encoder_layers,
            ).to(self.device)

        self.q_network = build_q()
        self.q_target = build_q()
        self.q_target.load_state_dict(self.q_network.state_dict())
        for parameter in self.q_target.parameters():
            parameter.requires_grad_(False)
        self.q_target.eval()

        self.actor = SharingActor(
            node_dim=NODE_FEATURE_DIM,
            hidden_units=config.hidden_units,
            selfishness=config.selfishness,
        ).to(self.device)

        # Published optimisers: RMSProp for the DGN low level, SGD for the
        # DDPG high level (traffic 6x6 column).
        self.q_optimizer = torch.optim.RMSprop(
            self.q_network.parameters(), lr=config.low_learning_rate
        )
        self.actor_optimizer = torch.optim.SGD(
            self.actor.parameters(), lr=config.high_learning_rate
        )
        self.replay = LatticeReplayBuffer(config.buffer_capacity)
        self.epsilon = config.epsilon_start
        self.updates = 0

        self.lattice: DistrictLattice | None = None
        self.sharing_weights: np.ndarray | None = None
        self.sharing_in: np.ndarray | None = None
        self.noise: OrnsteinUhlenbeckNoise | None = None
        self._env: UrbanLandUseEnv | None = None
        self._grid_before: ParcelGrid | None = None
        self._graph: GraphTensors | None = None
        self._pending: dict[str, np.ndarray] | None = None
        self._iteration = 0

    @classmethod
    def for_env(
        cls, env: UrbanLandUseEnv, seed: int | None = None, **overrides: object
    ) -> LToSMethod:
        """Build a method sized for ``env``'s grid."""
        return cls(LToSConfig(**overrides), seed=seed)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        return "ltos"

    # ------------------------------------------------------------------
    # episode control
    # ------------------------------------------------------------------
    def reset(self, env: UrbanLandUseEnv, seed: int | None = None) -> None:
        lattice = DistrictLattice.for_grid(
            env.grid.height, env.grid.width, self.config.districts_per_side
        )
        if self.lattice is not None and (
            lattice.n_districts != self.lattice.n_districts
            or lattice.district_size != self.lattice.district_size
        ):
            raise ValueError(
                "the district lattice changed shape; build a new LToS method "
                "for a different grid size"
            )
        self.lattice = lattice
        self._graph = GraphTensors.for_lattice(lattice, self.device)
        self._env = env
        self._grid_before = None
        self._pending = None
        self._iteration = 0
        self.sharing_weights = None
        self.noise = OrnsteinUhlenbeckNoise(
            size=lattice.n_districts * N_SLOTS,
            sigma=self.config.ou_sigma if self.config.learn_sharing else 0.0,
            rng=np.random.default_rng(seed),
        )

    # ------------------------------------------------------------------
    # acting
    # ------------------------------------------------------------------
    def decide(self, batch: DecisionBatch) -> BaselineDecision:
        if self._env is None or self.lattice is None:
            raise RuntimeError("call reset() before decide()")
        env = self._env
        self._grid_before = env.grid

        nodes = node_features(self.lattice, env.grid, env.attractiveness)
        # Appendix C's "different time scales": the high level acts every M
        # iterations and the low level reuses its weights in between.
        if self.sharing_weights is None or (
            self._iteration % self.config.high_action_interval == 0
        ):
            self.sharing_weights = self._sharing_action(nodes)
        self.sharing_in = incoming_weights(self.sharing_weights, self.lattice)

        parcels = parcel_features(env.grid, env.attractiveness, batch.parcels)
        districts = self.lattice.districts_of(batch.parcels)
        q_values = self._q_values(nodes, self.sharing_in, parcels, districts)

        greedy = q_values.argmax(axis=1)
        explore = self.rng.random(greedy.shape[0]) < self.epsilon
        intended = np.where(
            explore, self.rng.integers(0, 2, size=greedy.shape[0]), greedy
        ).astype(np.int64)

        executed = self._apply_budget(batch, intended, q_values)
        self._pending = {
            "node_features": nodes,
            "sharing_in": self.sharing_in,
            "parcel_features": parcels,
            "parcel_districts": districts,
            "actions": executed,
        }
        self._iteration += 1
        return BaselineDecision(
            decisions=np.where(executed == _CONVERT, APPROVE, REJECT).astype(np.int64)
        )

    def _apply_budget(
        self, batch: DecisionBatch, intended: np.ndarray, q_values: np.ndarray
    ) -> np.ndarray:
        """Keep only the top-B convert decisions, ranked by the Q-advantage."""
        assert self._env is not None
        advantage = q_values[:, _CONVERT] - q_values[:, _SKIP]
        wanted = intended == _CONVERT
        if not wanted.any():
            return np.zeros_like(intended)
        chosen = set(
            top_k_by_score(
                [parcel for parcel, flag in zip(batch.parcels, wanted) if flag],
                advantage[wanted],
                conversion_budget(self._env),
            )
        )
        return np.asarray(
            [_CONVERT if parcel in chosen else _SKIP for parcel in batch.parcels],
            dtype=np.int64,
        )

    def _sharing_action(self, nodes: np.ndarray) -> np.ndarray:
        """``w_i^out = phi_i(o_i) + exploration noise``, renormalised."""
        assert self.lattice is not None and self._graph is not None
        with torch.no_grad():
            weights = (
                self.actor(self._tensor(nodes)[None], self._graph.neighbour_mask)[0]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        if self.config.learn_sharing and self.config.ou_sigma > 0.0:
            assert self.noise is not None
            perturbation = self.noise.sample().reshape(
                self.lattice.n_districts, N_SLOTS
            )
            weights = np.clip(weights + perturbation, 0.0, None)
        weights = weights * self.lattice.neighbour_mask
        totals = weights.sum(axis=1, keepdims=True)
        # A district always keeps its self-loop, so the fallback is only reached
        # if noise wipes out an entire row; then share evenly over ``N_i``.
        fallback = self.lattice.neighbour_mask / self.lattice.neighbour_mask.sum(
            axis=1, keepdims=True
        )
        return np.where(
            totals > 0.0, weights / np.where(totals > 0.0, totals, 1.0), fallback
        )

    def _q_values(
        self,
        nodes: np.ndarray,
        sharing_in: np.ndarray,
        parcels: np.ndarray,
        districts: np.ndarray,
    ) -> np.ndarray:
        assert self._graph is not None
        with torch.no_grad():
            values = self.q_network(
                self._tensor(nodes)[None],
                self._tensor(sharing_in)[None],
                self._tensor(parcels)[None],
                torch.as_tensor(districts, dtype=torch.long, device=self.device)[None],
                self._graph.neighbour_index,
                self._graph.neighbour_mask,
            )
        return values[0].cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def observe(self, result: StepResult) -> None:
        if self._env is None or self.lattice is None or self._pending is None:
            raise RuntimeError("call decide() before observe()")
        assert self._grid_before is not None and self.sharing_weights is not None
        env = self._env

        local = district_rewards(
            self.lattice,
            self._grid_before,
            result.conversions,
            weights=env.weights,
            societal=result.societal_reward,
            config=env.config.reward,
        )
        shaped = shaped_rewards(local, self.sharing_weights, self.lattice)

        next_parcels = result.next_batch.parcels
        self.replay.push(
            LToSTransition(
                node_features=self._pending["node_features"],
                sharing_in=self._pending["sharing_in"].astype(np.float32),
                parcel_features=self._pending["parcel_features"],
                parcel_districts=self._pending["parcel_districts"],
                actions=self._pending["actions"],
                rewards=shaped.astype(np.float32),
                next_node_features=node_features(
                    self.lattice, env.grid, env.attractiveness
                ),
                # Design note: ``w_i^in'`` is the sharing vector in force at the
                # next step; the high level only moves every M iterations, so
                # that is the current one except on a high-level action step.
                next_sharing_in=self._pending["sharing_in"].astype(np.float32),
                next_parcel_features=parcel_features(
                    env.grid, env.attractiveness, next_parcels
                ),
                next_parcel_districts=self.lattice.districts_of(next_parcels),
                done=result.done,
            )
        )

    def train_step(self) -> float | None:
        ready = max(self.config.learning_starts, self.config.batch_size)
        if len(self.replay) < ready:
            return None
        assert self.lattice is not None and self._graph is not None
        batch = self.replay.sample(self.config.batch_size, self.rng)
        loss = update_low_level(
            self.q_network,
            self.q_target,
            self.q_optimizer,
            batch,
            self._graph,
            self.config,
            self.device,
        )
        soft_update(self.q_target, self.q_network, self.config.tau)
        self.epsilon = max(
            self.config.epsilon_end, self.epsilon * self.config.epsilon_decay
        )
        self.updates += 1
        if (
            self.config.learn_sharing
            and self.updates % self.config.high_update_interval == 0
        ):
            update_high_level(
                self.q_network,
                self.actor,
                self.actor_optimizer,
                batch,
                self._graph,
                self.lattice,
                self.config,
                self.device,
            )
        return loss

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(array, dtype=np.float32), dtype=torch.float32, device=self.device
        )


class FixedLToSMethod(LToSMethod):
    """The published "fixed LToS" ablation: sharing frozen at the initialiser.

    Yi et al. report it in all three of their scenarios, and it is nearly free:
    the high level never updates and never explores, so every district keeps the
    selfishness initialiser's weights for the whole run while the DGN low level
    still learns.
    """

    def __init__(self, config: LToSConfig, seed: int | None = None) -> None:
        super().__init__(replace(config, learn_sharing=False), seed=seed)

    @property
    def name(self) -> str:
        return "fixed_ltos"


def fixed_ltos(
    env: UrbanLandUseEnv, seed: int | None = None, **overrides: object
) -> FixedLToSMethod:
    """Convenience factory for the frozen-sharing ablation."""
    return FixedLToSMethod.for_env(env, seed=seed, **overrides)  # type: ignore[return-value]
