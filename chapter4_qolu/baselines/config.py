"""Frozen configuration objects for the three baselines.

Kept out of :mod:`chapter4_qolu.config` on purpose: that module holds the
*published* system's settings, while everything here belongs to the comparison
methods.  The same convention applies though - a value that a source states
explicitly is used verbatim, and every choice a source leaves open carries a
``Design note`` comment at the exact place it is made.

Sources:

* P-MADDPG - Pelcner et al. (2024), Chapter 3 (`thesis/overleaf/chapters/
  chapter3_paper1.tex`, Section "Property-based MADDPG" and its Algorithm 1);
* LToS - Yi, Li, Wang & Lu, *Learning to Share in Multi-Agent Reinforcement
  Learning*, arXiv:2112.08702 / NeurIPS 2022, via the research note
  ``chapter4_qolu/docs/baselines_lto_s_and_mitm_research.md``;
* MitM - Bone, Dragicevic & White (2011), *IJGIS* 25(5):717-737, which is
  paywalled: every mechanism below is INFERRED from the abstract and the
  paper's reference list (same research note).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chapter4_qolu.config import QOLUConfig, QOLURewardConfig

__all__ = ["LToSConfig", "MitMConfig", "PMADDPGConfig"]

#: The discount used by the published QOLU agent (Table 1).  All three
#: baselines inherit it so the comparison is not confounded by the horizon.
_PUBLISHED_GAMMA: float = QOLUConfig.gamma


@dataclass(frozen=True)
class PMADDPGConfig:
    """Property-based MADDPG hyper-parameters (Chapter 3).

    Published by Chapter 3: agents are static and act on their own cells;
    actions *and* rewards are modelled per cell; the multi-objective reward is
    linearised by a constant; a replay buffer ``Lambda`` is sampled uniformly
    for mini-batches ``{s, a, s', r}``; each agent has an actor ``upsilon_i``
    and a critic ``psi_i`` with target copies; the critic target is
    ``psi_i^t(s', upsilon_i^t(s')) * gamma + r`` trained under MSE, and the
    actor loss is ``-mean psi_i(s, upsilon_i(s))``.

    Chapter 3 publishes four optimisation numbers (chapter3_paper1.tex lines
    392-393): a discount ``gamma = 0.1``, target networks refreshed "every 250
    steps", an exploration coefficient ``epsilon = 0.3`` and a learning rate of
    ``1e-4`` "to overcome the issue of agents becoming stuck in local minima".
    All four are the defaults below.  Network sizes, buffer sizes and the batch
    size are *not* published for this environment - its own numbers belong to
    the Pollinators Game - so those carry a ``Design note`` on the field that
    sets them.

    Everything that is *not* Chapter 3's published value is reachable through
    :data:`chapter4_qolu.baselines.pmaddpg.PMADDPG_VARIANTS`; in particular
    ``variant="v1"`` selects Adam at 1e-3, ``gamma`` 0.99, Polyak averaging,
    Ornstein-Uhlenbeck exploration and no policy regulariser.
    """

    #: Land-cover planes plus the attractiveness plane (from the environment).
    observation_channels: int
    #: The published QOLU observation window, so both methods see the same thing.
    observation_window: int
    #: One P-MADDPG agent per top-down stakeholder (the paper runs three).
    n_agents: int = 3

    # --- architecture ---
    #: Design note: a compact convolutional encoder over the same observation
    #: patch QOLU uses.  Two layers rather than Table 1's three because the
    #: actor emits a single scalar per cell, not a 51-atom distribution.
    conv_channels: tuple[int, ...] = (32, 64)
    kernel_size: int = 3
    #: Design note: "same" padding, matching ``QOLUConfig.conv_padding``.
    conv_padding: int = 1
    #: Design note: one 256-unit hidden layer in both actor and critic heads.
    hidden_units: int = 256

    # --- optimisation ---
    #: PUBLISHED (Chapter 3 line 393): "We use a low learning rate value of
    #: $1e-4$, to overcome the issue of agents becoming stuck in local minima
    #: and struggling to learn new policies."  Design note: Chapter 3 quotes one
    #: rate and does not distinguish actor from critic, so both take it.
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    #: Polyak coefficient, used only when ``target_update_interval`` is ``None``
    #: (the thesis's own Appendix B table lists "Target network update rate
    #: tau = 0.01", which is the reading ``variant="v1"`` keeps).
    tau: float = 0.01
    #: PUBLISHED (Chapter 3 line 392): "We update the networks to the target
    #: networks every 250 steps", i.e. a *hard copy*, not Polyak averaging.
    #: Design note: "step" is read as one learning update of the agent, which is
    #: what Algorithm 1 iterates; ``None`` selects Polyak averaging at ``tau``.
    target_update_interval: int | None = 250
    #: PUBLISHED (Chapter 3 line 392): "The discount factor is $\\gamma = 0.1$."
    gamma: float = 0.1
    #: Design note: 256 transitions per mini-batch, matching Table 1's batch size
    #: so the two methods consume the same amount of experience per update.
    #: (The thesis's Appendix B independently lists a 256 mini-batch for the
    #: Pollinators Game, so the two readings agree.)
    batch_size: int = 256
    replay_capacity: int = 100_000
    learning_starts: int = 1_000
    #: Design note: gradient-norm clipping at 10, as for the QOLU agent.
    max_grad_norm: float | None = 10.0
    #: Design note (Chapter 3, Algorithm 1 line 6): the published target is
    #: ``psi^t * gamma + r`` with *no* terminal mask, so that is the default;
    #: ``True`` selects the standard ``(1 - done)`` masked form instead.
    mask_terminal: bool = False
    #: Design note (source: Lowe et al. 2017, the MADDPG paper this class already
    #: cites - its reference implementation adds ``1e-3 * mean(logits ** 2)`` to
    #: every actor loss).  Chapter 3 writes the actor loss as ``-mean
    #: psi_i(s, upsilon_i(s))`` alone; with the squashed actor this environment
    #: needs, that term alone leaves the pre-activation unbounded, so the first
    #: few updates drive it to |logit| 37-92, the sigmoid derivative underflows
    #: and the critic can no longer move the policy.  The penalty keeps the
    #: pre-activation in range; 0.0 restores the literal published loss
    #: (``variant="v1"``).
    policy_regulariser: float = 1e-3

    # --- action semantics ---
    #: The actor is continuous (DDPG), the environment is binary, so the
    #: approve-score is thresholded.  Design note: at 0.5, the midpoint of the
    #: actor's sigmoid range, so an untrained actor is unbiased.
    approve_threshold: float = 0.5

    # --- exploration ---
    #: PUBLISHED (Chapter 3 line 392): "Our exploration and exploitation
    #: $\\epsilon$ coefficient is equal to $0.3$."  Design note: Chapter 3 gives
    #: the coefficient but not its granularity, so it is applied per *candidate
    #: parcel* - the unit an action applies to in a property-based problem -
    #: and the random action of a continuous approve-score is a uniform draw
    #: from [0, 1].  ``None`` selects the Ornstein-Uhlenbeck alternative below.
    epsilon_greedy: float | None = 0.3
    #: Design note: DDPG's Ornstein-Uhlenbeck noise, decayed multiplicatively.
    #: Used only when ``epsilon_greedy`` is ``None``.
    ou_theta: float = 0.15
    ou_sigma: float = 0.2
    ou_dt: float = 1e-2
    noise_decay: float = 0.9995
    min_noise_scale: float = 0.05

    #: Which reading of ``R_QOLU,i`` pays the agents.  The default is the same
    #: per-parcel compliance signal the QOLU stakeholders learn from, so the
    #: comparison isolates the *algorithm*.
    qolu_reward: QOLURewardConfig = field(default_factory=QOLURewardConfig)
    #: Chapter 3's linearisation constant ``alpha``, translated into this
    #: environment's mixing form.  ``None`` leaves ``qolu_reward`` untouched;
    #: any float *overrides* ``qolu_reward.team_reward_weight`` for P-MADDPG
    #: only, so a campaign that hands the method its own ``QOLURewardConfig``
    #: (``scenario_worker._build_baseline`` does) cannot silently erase it.
    #:
    #: Design note (the 0.02): Chapter 3 line 139 defines ``R_i = alpha_i *
    #: R_local,i + (1 - alpha_i) * R_global + R_inc,i / 2`` with ``R_local,i =
    #: sum_{c in C_i} F(c)`` over ``M = |C_i|`` cells, whereas
    #: ``env.qolu_rewards.qolu_parcel_rewards`` mixes ``F(c) + w * R_team``
    #: *per cell*.  Dividing Chapter 3's reward by ``alpha`` (a positive
    #: constant, so the arg-max is unchanged) and spreading the global term
    #: evenly over the ``M`` cells of the property gives ``F(c) + w * R_global``
    #: with ``w = (1 - alpha) / (alpha * M)``.  Chapter 3 publishes no single
    #: ``alpha`` - it is an experimental axis, the HM sweep ``{0.0, 0.2, 0.4,
    #: 0.6, 0.8, 1.0}`` (line 373) - so the symmetric centre of that axis,
    #: ``alpha = 0.5`` (local and global valued equally), is used, and ``M =
    #: 50`` is Chapter 3's own study-vector size (line 392), which happens to
    #: equal this environment's ``EnvConfig.samples_per_iteration``.  Hence
    #: ``w = 0.5 / (0.5 * 50) = 0.02``.  Two departures remain and are *not*
    #: forced: Chapter 4 has no regulatory agent, so ``R_inc,i / 2`` has no
    #: counterpart, and the convex form is reproduced only up to the positive
    #: factor ``alpha``, which rescales Q-values but not the policy.
    team_reward_weight: float | None = 0.02
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.observation_channels < 1:
            raise ValueError("observation_channels must be positive")
        if self.observation_window < 1 or self.observation_window % 2 == 0:
            raise ValueError("observation_window must be a positive odd number")
        if self.n_agents < 1:
            raise ValueError("n_agents must be positive")
        if not self.conv_channels:
            raise ValueError("at least one convolution layer is required")
        if self.hidden_units < 1:
            raise ValueError("hidden_units must be positive")
        if self.actor_learning_rate <= 0.0 or self.critic_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must lie in (0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least the batch size")
        if self.learning_starts < 1:
            raise ValueError("learning_starts must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive when given")
        if self.target_update_interval is not None and self.target_update_interval < 1:
            raise ValueError("target_update_interval must be positive or None")
        if self.policy_regulariser < 0.0:
            raise ValueError("policy_regulariser must be non-negative")
        if not 0.0 <= self.approve_threshold <= 1.0:
            raise ValueError("approve_threshold must lie in [0, 1]")
        if self.epsilon_greedy is not None and not 0.0 <= self.epsilon_greedy <= 1.0:
            raise ValueError("epsilon_greedy must lie in [0, 1] or be None")
        if not 0.0 < self.noise_decay <= 1.0:
            raise ValueError("noise_decay must lie in (0, 1]")
        if not 0.0 <= self.min_noise_scale <= 1.0:
            raise ValueError("min_noise_scale must lie in [0, 1]")
        if self.team_reward_weight is not None and self.team_reward_weight < 0.0:
            raise ValueError("team_reward_weight must be non-negative or None")


@dataclass(frozen=True)
class LToSConfig:
    """LToS hyper-parameters (Yi et al. 2021), traffic 6x6 column by default.

    Published by Yi et al.: a low level learned with DGN over discrete actions
    and conditioned on the incoming sharing weights ``w_i^in``; a high level
    learned with DDPG whose softmax output is the outgoing sharing vector
    ``w_i^out``; the shaped reward ``r_i^w = sum_{j in N_i} w_ji r_j``; and the
    three Appendix-C training tricks (selfishness initialiser, shared-seed
    synchronised replay sampling, and different time scales for the two levels).

    The numeric defaults are the published traffic 6x6 column - the closest
    analogue to a fixed 2-D lattice of spatial agents - *except* ``gamma``,
    which is taken from the published Chapter 4 configuration instead of the
    traffic value 0.8, because one run here is a single 1000-iteration episode.
    """

    #: 50 / 10 = 5, i.e. the note's recommended 10 x 10 lattice of 5 x 5 parcels.
    districts_per_side: int = 10
    gamma: float = _PUBLISHED_GAMMA

    # --- low level (DGN), traffic 6x6 column ---
    buffer_capacity: int = 10_000
    #: "sample size / batch = 1000 / 20".  Design note: that is DGN's practice of
    #: drawing a large sample and iterating mini-batches of 20 over it; here one
    #: mini-batch of 20 whole-lattice timesteps is drawn per update, which keeps
    #: the per-iteration cost bounded.
    batch_size: int = 20
    learning_starts: int = 20
    #: RMSProp at 1e-3 with soft target updates at tau = 0.1 (published).
    low_learning_rate: float = 1e-3
    tau: float = 0.1
    #: "MLP units / activation: 32 & 32, ReLU" (published).
    hidden_units: int = 32
    #: Design note: standard DGN defaults (Jiang et al., ICLR 2020) rather than
    #: the values of Yi et al.'s Table 4: a single encoder layer, two
    #: graph-convolution layers and eight attention heads.
    encoder_layers: int = 1
    attention_layers: int = 2
    attention_heads: int = 8
    #: "eps start / decay / end = 0.4 / 0.9 / 0.05" (published); Design note: the
    #: decay is multiplicative and applied once per learning update.
    epsilon_start: float = 0.4
    epsilon_decay: float = 0.9
    epsilon_end: float = 0.05
    max_grad_norm: float | None = 10.0

    # --- high level (DDPG), traffic 6x6 column ---
    #: SGD at 1e-3 with Ornstein-Uhlenbeck noise, sigma = 0.25 (published).
    high_learning_rate: float = 1e-3
    ou_sigma: float = 0.25
    #: "update frequency: every 20 episodes".  Design note: a run here is a
    #: single episode, so the published *episode* counts are read as iteration
    #: counts - the high level updates every 20 iterations.
    high_update_interval: int = 20
    #: "action interval M = 15 steps" (published): the high level re-issues
    #: ``w^out`` every 15 iterations and the low level reuses it in between.
    high_action_interval: int = 15
    #: "initialiser: selfishness, 0.8" (published, Appendix C).
    selfishness: float = 0.8
    #: ``False`` freezes the sharing weights at the initialiser - the published
    #: "fixed LToS" ablation, which Yi et al. report in all three scenarios.
    learn_sharing: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.districts_per_side < 1:
            raise ValueError("districts_per_side must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if self.buffer_capacity < 1:
            raise ValueError("buffer_capacity must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_starts < 1:
            raise ValueError("learning_starts must be positive")
        if self.low_learning_rate <= 0.0 or self.high_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must lie in (0, 1]")
        if self.hidden_units < 1:
            raise ValueError("hidden_units must be positive")
        if self.hidden_units % self.attention_heads != 0:
            raise ValueError("hidden_units must be divisible by attention_heads")
        if self.encoder_layers < 1 or self.attention_layers < 1:
            raise ValueError("encoder_layers and attention_layers must be positive")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("require 0 <= epsilon_end <= epsilon_start <= 1")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must lie in (0, 1]")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive when given")
        if self.ou_sigma < 0.0:
            raise ValueError("ou_sigma must be non-negative")
        if self.high_update_interval < 1 or self.high_action_interval < 1:
            raise ValueError("high-level intervals must be positive")
        if not 0.0 < self.selfishness < 1.0:
            raise ValueError("selfishness must lie in (0, 1)")


@dataclass(frozen=True)
class MitMConfig:
    """MitM parameters - **every one of them is INFERRED**.

    Bone, Dragicevic & White (2011) is paywalled and this implementation
    follows its abstract and reference list, so there are no published values
    at all: not the learning rate, not the number of stakeholders, not the grid
    size, not the number of rounds.  The implementation and its sources are
    documented in
    :mod:`chapter4_qolu.baselines.mitm_allocation` and
    :mod:`chapter4_qolu.baselines.mitm`.

    The defaults are chosen to be *weak and transparent* rather than tuned: MitM
    is a 2011 GIS/ABM method and silently strengthening it would misrepresent
    the comparison.
    """

    #: ``lambda`` of ``A = s * (1 + lambda * N) * (1 + eps)``.  Design note: 1.0,
    #: i.e. the neighbourhood term can at most double the intrinsic suitability.
    neighbourhood_weight: float = 1.0
    #: Half-width of the uniform stochastic perturbation ``eps`` that White &
    #: Engelen add to every cell.  Design note: 0.1.
    perturbation_scale: float = 0.1
    #: Strength of the population-density discount.  Design note: 1.0.
    density_pressure: float = 1.0
    #: How many cells the bottom-up layer puts forward, as a multiple of the
    #: shared conversion budget.  Design note: 2, so the MODM step has a real
    #: choice to resolve; exogenous demand is otherwise the budget itself,
    #: because ``WellBeingConfig`` keeps the population static by default.
    proposal_multiplier: int = 2
    #: ``p`` of the compromise-programming metric.  Design note: 2 (Euclidean).
    compromise_order: float = 2.0
    #: ``False`` (the headline variant) freezes ``theta`` at the scenario
    #: weights so that the four-scenario axis keeps its meaning; ``True`` is the
    #: secondary variant in which MitM negotiates the weights endogenously,
    #: which is truer to MitM but makes the scenario axis drift.
    learn_weights: bool = False
    #: ``alpha`` of the TD(0) update ``V <- V + alpha (R - V)``.  Design note:
    #: 0.1, a slow tabular rate.
    value_learning_rate: float = 0.1
    #: How far the realised criterion profile pulls ``theta``.  Design note: 0.05.
    weight_learning_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.neighbourhood_weight < 0.0:
            raise ValueError("neighbourhood_weight must be non-negative")
        if self.perturbation_scale < 0.0:
            raise ValueError("perturbation_scale must be non-negative")
        if self.density_pressure < 0.0:
            raise ValueError("density_pressure must be non-negative")
        if self.proposal_multiplier < 1:
            raise ValueError("proposal_multiplier must be at least one")
        if self.compromise_order <= 0.0:
            raise ValueError("compromise_order must be positive")
        if not 0.0 <= self.value_learning_rate <= 1.0:
            raise ValueError("value_learning_rate must lie in [0, 1]")
        if not 0.0 <= self.weight_learning_rate <= 1.0:
            raise ValueError("weight_learning_rate must lie in [0, 1]")
