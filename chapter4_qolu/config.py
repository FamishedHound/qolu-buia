"""Frozen configuration objects for the Chapter 4 package.

Every value that the paper states explicitly is used as the default here.
Values the paper leaves open carry a ``Design note`` comment giving the
reasoning behind the choice, so the difference between a published value and
a modelling decision stays visible in the code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from chapter4_qolu.land_cover import LandCoverGroup
from chapter4_qolu.objectives import TopDownObjective

__all__ = [
    "NOISY_SCOPES",
    "SCENARIOS",
    "AttractivenessConfig",
    "BUIAConfig",
    "EnvConfig",
    "QOLUConfig",
    "QOLURewardConfig",
    "RewardConfig",
    "ScenarioWeights",
    "WellBeingConfig",
]

#: Which fully-connected layers of the QOLU network carry NoisyNet noise.
#:
#: Design note (scope): Hessel et al. (2018) "replace all linear layers with
#: their noisy equivalent"; in Rainbow that is each stream's hidden layer plus
#: its output layer.  QOLU's Table 1 trunk - the projection and the residual
#: MLP - is *shared* by the two streams and has no Rainbow analogue, so both
#: readings are kept: ``all_linear`` is the literal one and the default,
#: ``streams`` restricts the noise to the value and advantage heads.
#:
#: Declared here rather than in :mod:`chapter4_qolu.agent.noisy` so that this
#: module stays importable without torch.
NOISY_SCOPES: tuple[str, ...] = ("all_linear", "streams")


@dataclass(frozen=True)
class ScenarioWeights:
    """Stakeholder weights ``w_i`` of the reward ``sum_i w_i R_phi_i + R_society``.

    The three components are the paper's result-table columns (Table 2).
    """

    land_use: float
    woodland: float
    urban: float

    def __post_init__(self) -> None:
        for name in ("land_use", "woodland", "urban"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(
                    f"scenario weight {name} must be non-negative, got {value}"
                )

    def __getitem__(self, objective: TopDownObjective) -> float:
        return {
            TopDownObjective.LAND_USE_SEPARATION: self.land_use,
            TopDownObjective.WOODLAND_PRESERVATION: self.woodland,
            TopDownObjective.URBAN_PROXIMITY: self.urban,
        }[objective]

    def total(self) -> float:
        """Sum of the three weights (0.99 for the published Scenario 1)."""
        return float(self.land_use + self.woodland + self.urban)

    def normalised(self) -> ScenarioWeights:
        """Return the same weighting rescaled to sum to one."""
        total = self.total()
        if total <= 0.0:
            raise ValueError("cannot normalise an all-zero weighting")
        return ScenarioWeights(
            land_use=self.land_use / total,
            woodland=self.woodland / total,
            urban=self.urban / total,
        )

    def as_mapping(self) -> Mapping[TopDownObjective, float]:
        return {objective: self[objective] for objective in TopDownObjective}


#: The four published preference settings (Table 2 of the paper).  Scenario 1's
#: weights sum to 0.99 rather than 1.0 because the paper rounds 1/3 to 0.33;
#: :meth:`ScenarioWeights.normalised` is available where that matters.
SCENARIOS: Mapping[int, ScenarioWeights] = {
    1: ScenarioWeights(land_use=0.33, woodland=0.33, urban=0.33),
    2: ScenarioWeights(land_use=0.50, woodland=0.25, urban=0.25),
    3: ScenarioWeights(land_use=0.25, woodland=0.50, urban=0.25),
    4: ScenarioWeights(land_use=0.25, woodland=0.25, urban=0.50),
}


@dataclass(frozen=True)
class RewardConfig:
    """Parameters of the reward terms ``R_phi_i`` and ``R_society``.

    ``alpha`` and ``beta`` are named in the paper ("the societal reward captures
    the net public benefit of protecting overall forest and agricultural areas,
    parameterized by alpha and beta") but no values are given.

    Design note: alpha = beta = 0.5, i.e. forest and agricultural land are
    equally valuable to society and the societal term is bounded in [0, 1] like
    the three stakeholder terms.
    """

    alpha: float = 0.5
    beta: float = 0.5
    #: Land types a QOLU stakeholder wants to keep out of development.  The
    #: paper's experimental settings say "woodland and arable land"; switch the
    #: second entry to ``LandCoverGroup.AGRICULTURAL`` for the methodology
    #: section's broader "agricultural land" reading.
    protected_groups: tuple[LandCoverGroup, ...] = (
        LandCoverGroup.WOODLAND,
        LandCoverGroup.ARABLE,
    )
    #: Habitat that development should stay away from (objective (i)).
    separation_group: LandCoverGroup = LandCoverGroup.MOUNTAIN_HEATH_BOG
    #: Land types whose *total area* the societal term protects.
    societal_forest_group: LandCoverGroup = LandCoverGroup.WOODLAND
    societal_agricultural_group: LandCoverGroup = LandCoverGroup.AGRICULTURAL
    #: A parcel counts as "being" a land type when its cover fraction reaches
    #: this threshold.  Design note: 0.5, i.e. the dominant-cover reading of the
    #: 1 km percentage product.
    presence_threshold: float = 0.5
    #: Distance used to normalise the two proximity terms into [0, 1].
    #: ``None`` means "the diagonal of the current grid".
    max_distance: float | None = None

    def __post_init__(self) -> None:
        if self.alpha < 0.0 or self.beta < 0.0:
            raise ValueError("alpha and beta must be non-negative")
        if not 0.0 < self.presence_threshold <= 1.0:
            raise ValueError("presence_threshold must lie in (0, 1]")
        if self.max_distance is not None and self.max_distance <= 0.0:
            raise ValueError("max_distance must be positive when given")
        if not self.protected_groups:
            raise ValueError("at least one protected group is required")


@dataclass(frozen=True)
class AttractivenessConfig:
    """Parameters of the attractiveness metric ``B_attractiveness`` (Eq. 1).

    The paper defines attractiveness qualitatively: "a score based on proximity
    to natural features (e.g. forests, fluvial areas) and existing urban
    centres", updated after each conversion "to reflect changes in neighbouring
    parcels".

    Design note: the score of parcel ``i`` is a distance-decayed, neighbourhood
    normalised average of the three feature groups,

        A_i = sum_g w_g * ( sum_{j in N_r(i)} k(d_ij) f_g(j) ) / sum_j k(d_ij)

    with ``k(d) = exp(-d / decay_length)`` and ``f_g(j)`` the cover fraction of
    group ``g`` in parcel ``j``.  This keeps ``A_i`` in [0, 1] and makes the
    post-conversion update strictly local, matching the paper's description.

    Design note: equal weight on the three named feature groups (woodland,
    freshwater, urban) and ``radius = 5`` parcels (5 km) with a 2 km decay
    length, which is the same order of magnitude as BUIA's stated 2 km radius.
    """

    radius: int = 5
    decay_length: float = 2.0
    group_weights: tuple[tuple[LandCoverGroup, float], ...] = (
        (LandCoverGroup.WOODLAND, 1.0 / 3.0),
        (LandCoverGroup.FRESHWATER, 1.0 / 3.0),
        (LandCoverGroup.URBAN, 1.0 / 3.0),
    )

    def __post_init__(self) -> None:
        if self.radius < 1:
            raise ValueError("radius must be at least one parcel")
        if self.decay_length <= 0.0:
            raise ValueError("decay_length must be positive")
        if not self.group_weights:
            raise ValueError("at least one feature group is required")
        total = 0.0
        for group, weight in self.group_weights:
            if weight < 0.0:
                raise ValueError(f"weight for {group} must be non-negative")
            total += weight
        if total <= 0.0:
            raise ValueError("feature group weights must not sum to zero")


@dataclass(frozen=True)
class QOLURewardConfig:
    """How a stakeholder's transition reward is built and how its vote counts.

    Section 3.1 vs 3.5.

    The paper describes two different things and never reconciles them:
    Section 3.1 has the environment supply "one shared team reward", while
    Section 3.5 has each QOLU agent allocate rewards "based on the degree of
    compliance with their individual objectives: R_QOLU,i = f(parcel_data,
    objectives_i)".  All three readable variants are available:

    * ``"per_parcel"`` (default) - Section 3.5 taken literally, with
      ``parcel_data`` read as the *single* candidate parcel being appraised, so
      the reward depends on the parcel *and* on the action taken;
    * ``"objective"`` - Section 3.5 at iteration level: the stakeholder's own
      objective reward ``R_phi_i`` for the whole synchronised update;
    * ``"team"`` - Section 3.1: the shared ``sum_i w_i R_phi_i + R_society``.

    Design note (``approval_form``): the default ``"symmetric"`` form pays
    ``+(compliance - 0.5)`` for APPROVE and ``-(compliance - 0.5)`` for REJECT.
    Rejecting a non-compliant parcel is therefore rewarded, which is what makes
    a guardian policy learnable at all: under the ``"asymmetric"`` alternative
    (APPROVE -> ``compliance - 0.5``, REJECT -> 0) every rejection looks
    identical, so the agent gets no signal separating a good refusal from a bad
    one.  The 0.5 pivot makes the reward zero-mean under a random policy, which
    keeps returns centred on the published [-200, 200] atom support.

    Design note (scope): a stakeholder is paid for the *appraisal it made*, not
    for whether the conversion actually happened - the paper's compliance
    wording is about the judgement.  Three cases, all deliberate:
    approved-and-converted, proposed-but-not-converted (consensus or the
    per-iteration cap blocked it), and never-proposed parcels all earn the same
    formula.  ``unproposed_reward_scale`` below down-weights (0.0 removes) the
    third case for anyone who wants the signal restricted to parcels that were
    genuinely in play.

    Design note (``team_reward_weight``): 0.0 by default, i.e. the Section 3.5
    reading is used on its own; a positive value adds
    ``weight * team_reward`` to every transition to mix Section 3.1's
    cooperative term back in.

    Where the *scenario weights* enter
    ---------------------------------
    Under the defaults below they do not enter a top-down agent at all: the
    per-parcel reward of stakeholder ``i`` depends only on compliance with its
    own objective, and ``agent.consensus`` sums the quality scores unweighted.
    ``w_i`` then reaches nothing but the reported team reward, so QOLU + BUIA
    (and P-MADDPG, which shares both code paths) produces identical
    trajectories for every scenario whose weights it is run under.  The paper
    reads the weights as per-stakeholder *preference* weights - ``R = sum_i w_i
    R_phi_i + R_society`` (Section 3.1), ``LP = sum_i w_i P_phi_TD_i`` (Eq. 3) -
    and reports a different winner per scenario, so two optional readings are
    provided.  Both are OFF by default: turning either on changes results, and
    the published reading is the unweighted one.

    Design note (``scale_by_scenario_weight``): the per-parcel reading of
    ``sum_i w_i R_phi_i``.  Stakeholder ``i``'s appraisal reward is multiplied
    by ``n_stakeholders * w_i`` (the weighting first normalised to sum to one,
    so the multipliers average exactly 1 and the overall reward scale - and
    with it the published [-200, 200] atom support - is unaffected by which
    scenario is run).  Because the learned Q-value *magnitudes* then scale with
    ``w_i``, the paper's plain-sum consensus becomes effectively weighted
    without changing the consensus rule itself.  The shared team term added by
    ``team_reward_weight`` is deliberately left unscaled: it already carries the
    weights inside ``sum_i w_i R_phi_i + R_society``.  For the same reason the
    switch applies to ``mode="per_parcel"`` only and is deliberately inert in
    the other two: ``"team"`` already *is* ``sum_i w_i R_phi_i + R_society``, so
    weighting it again would square the preferences, and ``"objective"`` is left
    exactly as published rather than given a second, untested weighting.

    Design note (``weighted_consensus``): the alternative reading, in which the
    weights are the balancing instrument of the consensus rule itself -
    "Summing these scores across all stakeholders provides a consensus-driven
    measure that balances ecological, economic, and policy-related objectives".
    Each stakeholder's contribution to the sum is multiplied by its scenario
    weight, normalised to mean 1 so that equal weights reproduce the published
    plain sum exactly.  Independent of the switch above; either, both or
    neither may be used.
    """

    mode: str = "per_parcel"
    approval_form: str = "symmetric"
    team_reward_weight: float = 0.0
    unproposed_reward_scale: float = 1.0
    #: Compliance value that separates "worth approving" from "worth rejecting".
    neutral_compliance: float = 0.5
    #: Pay stakeholder ``i`` ``n_stakeholders * w_i`` times its own reward.
    scale_by_scenario_weight: bool = False
    #: Weight each stakeholder's contribution to the consensus sum by ``w_i``.
    weighted_consensus: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("per_parcel", "objective", "team"):
            raise ValueError(
                "mode must be 'per_parcel', 'objective' or 'team', "
                f"got {self.mode!r}"
            )
        if self.approval_form not in ("symmetric", "asymmetric"):
            raise ValueError(
                f"approval_form must be 'symmetric' or 'asymmetric', "
                f"got {self.approval_form!r}"
            )
        if self.team_reward_weight < 0.0:
            raise ValueError("team_reward_weight must be non-negative")
        if not 0.0 <= self.unproposed_reward_scale <= 1.0:
            raise ValueError("unproposed_reward_scale must lie in [0, 1]")
        if not 0.0 <= self.neutral_compliance <= 1.0:
            raise ValueError("neutral_compliance must lie in [0, 1]")


@dataclass(frozen=True)
class WellBeingConfig:
    """Housing dynamics behind the well-being metric ``WB = H_occupied / H_total``.

    Section 3.6 defines the metric but the paper never states how housing
    capacity or occupancy are produced, so the whole of this configuration is a
    modelling decision; it is kept as small as possible.

    Design note: a parcel's housing capacity is ``capacity_per_parcel`` times its
    built-up cover fraction, i.e. capacity is created by land conversion and
    nothing else.

    Design note: households pick parcels in proportion to attractiveness (the
    abstract's "dynamic population influx in different regions"), in a single
    allocation pass - demand that exceeds a parcel's capacity is lost rather
    than re-housed elsewhere, so spatial mismatch shows up as a lower WB.

    Design note: the starting population is ``initial_occupancy`` times the
    capacity of the *initial* landscape and grows linearly at
    ``population_growth_per_iteration`` per iteration (0 by default, matching
    the paper's "relatively static socio-economic conditions" limitation).
    """

    capacity_per_parcel: float = 1000.0
    initial_occupancy: float = 0.9
    population_growth_per_iteration: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity_per_parcel <= 0.0:
            raise ValueError("capacity_per_parcel must be positive")
        if not 0.0 <= self.initial_occupancy <= 1.0:
            raise ValueError("initial_occupancy must lie in [0, 1]")
        if self.population_growth_per_iteration < 0.0:
            raise ValueError("population_growth_per_iteration must be non-negative")


@dataclass(frozen=True)
class EnvConfig:
    """Environment settings (Section 3.5 and the experimental settings).

    Published values: 50 parcels sampled with replacement per iteration,
    1000 iterations per run, 1 km^2 parcels on a 50 x 50 grid (thesis text).

    ``grid_size``: the published paper's Figure 1 caption instead describes
    "blocks of 40 x 40 pixel units"; at the 25 m resolution of the LCM2015
    raster, 40 x 40 pixels *is* one 1 km^2 parcel, so the two statements are
    consistent and only the extent differs.  The grid size therefore stays a
    parameter, defaulting to the thesis's 50 x 50.

    ``max_conversions_per_iteration``: the paper says an iteration allows "the
    potential conversion of up to 11 land types into urban areas" while the
    thesis says 10.  Treated as a non-critical published inconsistency; the cap
    defaults to 10 and is configurable.

    Design note: ``max_conversions_per_iteration=None`` means *no* cap, i.e.
    every approved candidate converts.  The research note for the baselines
    points out that the published text never actually states a per-iteration
    budget, so "no budget" has to be expressible; the cap is one of the shared
    scale settings every method in :mod:`chapter4_qolu.baselines` runs under.
    """

    grid_size: int = 50
    #: Columns of the parcel grid; ``None`` means "square", i.e. ``grid_size``.
    #: Design note (rectangular extents): the paper's Figure 1 shows the whole
    #: South East England region at 1 km while the text says "a 50 x 50 grid of
    #: parcels, as shown in Figure 1".  The two cannot both hold, so the extent
    #: has to be expressible either way; ``grid_size`` alone keeps the thesis's
    #: square reading and is the default, and ``grid_width`` opens the
    #: whole-region one without changing any existing configuration.
    grid_width: int | None = None
    samples_per_iteration: int = 50
    max_conversions_per_iteration: int | None = 10
    iterations: int = 1000
    #: Size of the square neighbourhood a top-down agent observes around a
    #: candidate parcel.  Design note: the paper only says agents are partially
    #: observing; 9 km x 9 km is wide enough for the 3-layer 3x3 convolution
    #: stack of Table 1 to see context beyond BUIA's 2 km radius.
    observation_window: int = 9
    #: Size of the square neighbourhood a *bottom-up* investor observes.
    #: The paper constrains BUIA to "a 2 km radius"; at 1 km^2 parcels that is a
    #: radius of two parcels, i.e. a 5 x 5 window (documented reading).
    buia_observation_window: int = 5
    #: Share of a parcel's non-urban cover moved to built-up on conversion.
    #: Design note: 1.0 - the paper converts whole parcels ("convert selected
    #: parcels" / "non-residential parcel").
    conversion_fraction: float = 1.0
    #: A parcel stops being a development candidate once this much of it is
    #: already built up.
    urban_saturation: float = 0.99
    reward: RewardConfig = field(default_factory=RewardConfig)
    attractiveness: AttractivenessConfig = field(default_factory=AttractivenessConfig)
    well_being: WellBeingConfig = field(default_factory=WellBeingConfig)

    @property
    def grid_shape(self) -> tuple[int, int]:
        """``(rows, columns)`` of the parcel grid this configuration expects."""
        return int(self.grid_size), int(
            self.grid_size if self.grid_width is None else self.grid_width
        )

    def __post_init__(self) -> None:
        if self.grid_size < 1:
            raise ValueError("grid_size must be positive")
        if self.grid_width is not None and self.grid_width < 1:
            raise ValueError("grid_width must be positive")
        if self.samples_per_iteration < 1:
            raise ValueError("samples_per_iteration must be positive")
        if (
            self.max_conversions_per_iteration is not None
            and self.max_conversions_per_iteration < 0
        ):
            raise ValueError(
                "max_conversions_per_iteration must be non-negative or None"
            )
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.observation_window < 1 or self.observation_window % 2 == 0:
            raise ValueError("observation_window must be a positive odd number")
        if self.buia_observation_window < 1 or self.buia_observation_window % 2 == 0:
            raise ValueError("buia_observation_window must be a positive odd number")
        if not 0.0 < self.conversion_fraction <= 1.0:
            raise ValueError("conversion_fraction must lie in (0, 1]")
        if not 0.0 < self.urban_saturation <= 1.0:
            raise ValueError("urban_saturation must lie in (0, 1]")


@dataclass(frozen=True)
class QOLUConfig:
    """QOLU agent hyper-parameters.

    Published in Table 1: 3x3 convolutions with 32 -> 64 -> 128 filters, a
    residual MLP of two 256-unit layers, 51 atoms over [-200, 200], Adam,
    mini-batch 256, gamma = 0.99 and epsilon-greedy exploration with an
    unspecified schedule.

    Everything else follows standard Rainbow/C51 practice and carries a
    ``Design note`` on the field below.
    """

    observation_channels: int
    observation_window: int
    #: QOLU appraises a proposed conversion, so the action set is
    #: {reject, approve} (see :mod:`chapter4_qolu.env.urban_env`).
    n_actions: int = 2

    # --- architecture (Table 1) ---
    conv_channels: tuple[int, ...] = (32, 64, 128)
    kernel_size: int = 3
    #: Design note: padded ("same") convolutions so that small observation
    #: windows survive three stacked 3x3 layers.
    conv_padding: int = 1
    hidden_units: int = 256
    n_atoms: int = 51
    v_min: float = -200.0
    v_max: float = 200.0

    # --- optimisation (Table 1 where published) ---
    batch_size: int = 256
    gamma: float = 0.99
    #: Design note: Adam settings of Bellemare et al. (2017), whose atom support
    #: the paper adopts: lr 2.5e-4 with eps = 0.01 / batch_size.
    learning_rate: float = 2.5e-4
    adam_eps: float = 0.01 / 256
    #: Design note: Algorithm 1 uses an n-step return but does not fix n; 3 is
    #: the Rainbow default.
    n_step: int = 3
    replay_capacity: int = 100_000
    learning_starts: int = 1_000
    #: "Every t_n steps reset theta^- = theta"; t_n is not given.
    #: Design note: 25 gradient steps, so the reset fires on about 4 % of the
    #: ~980 gradient steps of a 1000-iteration run (39 times per stakeholder).
    #: The published text leaves t_n unspecified.
    target_sync_interval: int = 25
    #: How many ``learn()`` calls one environment iteration fires, i.e. the
    #: replay ratio of gradient steps to synchronised updates.
    #:
    #: Design note: 1, i.e. one ``learn()`` per iteration in the loops of
    #: :mod:`chapter4_qolu.runner` - Algorithm 1 interleaves
    #: "sample a minibatch ... perform a gradient descent step" with the actor
    #: loop and never says how many times, so one per iteration is the literal
    #: reading and stays the default.  It is a field rather than a constant
    #: because the consequence is easy to miss: at the campaign's 1000
    #: iterations that is ~980 gradient steps of batch 256 against 50 000
    #: stored transitions, so each transition is sampled only about five times
    #: in a whole run, and whether that is enough to learn a state-conditional
    #: policy is an experiment rather than an opinion.  Raising it multiplies
    #: the gradient budget without touching the environment, the replay
    #: contents or the reward.
    learn_steps_per_iteration: int = 1
    #: Design note: gradient-norm clipping at 10 (Rainbow default), for
    #: stability with the wide [-200, 200] atom support.
    max_grad_norm: float = 10.0

    # --- exploration (Table 1: "epsilon-greedy (schedule not fixed)") ---
    #: Design note: linear decay 1.0 -> 0.05.  Algorithm 1 line 5 keeps *both*
    #: mechanisms - "exploiting noisy network parameters theta **or**
    #: exploratory action based on Epsilon" - so epsilon-greedy is retained on
    #: top of the NoisyNet layers rather than replaced by them, which is also
    #: the only reading consistent with Table 1 listing epsilon-greedy as the
    #: exploration row.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000

    # --- Rainbow components (Hessel et al. 2018) ---
    #: NoisyNet layers (Fortunato et al. 2018).  **Published**: Algorithm 1
    #: line 5 selects an action "by exploiting noisy network parameters
    #: theta", so they are on by default.  ``False`` selects ordinary linear
    #: layers, reachable as the ``qolu_c51`` method variant.
    noisy_nets: bool = True
    #: Design note: sigma_0 = 0.5, the factorised-noise value of Fortunato et
    #: al. (2018) that Rainbow adopts.  The paper states none.
    noisy_sigma0: float = 0.5
    #: Design note: "all linear layers", as Hessel et al. put it - the shared
    #: projection, the residual MLP and both dueling heads.  ``"streams"``
    #: restricts the noise to the value and advantage heads.
    noisy_scope: str = "all_linear"
    #: Prioritised experience replay (Schaul et al. 2016, proportional).
    #: **Off by default**: the chapter never mentions prioritisation, and
    #: Algorithm 1 says only "sample a minibatch of transitions from D", so
    #: uniform sampling is the published reading.  Available as the
    #: ``qolu_buia_per`` method variant, an opt-in Rainbow extra.
    prioritised_replay: bool = False
    #: Design note: Rainbow's published PER settings - alpha 0.5 and beta
    #: annealed 0.4 -> 1.0 over the run.  The paper states none of them.
    per_alpha: float = 0.5
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    #: Design note: "over the run" is read as the campaign's 1000 iterations,
    #: which is one gradient step each; a longer run should raise this.
    per_beta_steps: int = 1_000
    #: Design note: the usual guard keeping a zero-KL transition resampleable.
    per_epsilon: float = 1e-6

    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.observation_channels < 1:
            raise ValueError("observation_channels must be positive")
        if self.observation_window < 1 or self.observation_window % 2 == 0:
            raise ValueError("observation_window must be a positive odd number")
        if self.n_actions < 2:
            raise ValueError("n_actions must be at least two (reject / approve)")
        if not self.conv_channels:
            raise ValueError("at least one convolution layer is required")
        if self.hidden_units < 1:
            raise ValueError("hidden_units must be positive")
        if self.n_atoms < 2:
            raise ValueError("n_atoms must be at least two")
        if self.v_max <= self.v_min:
            raise ValueError("v_max must be greater than v_min")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.n_step < 1:
            raise ValueError("n_step must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least the batch size")
        if self.learning_starts < 1:
            raise ValueError("learning_starts must be positive")
        if self.target_sync_interval < 1:
            raise ValueError("target_sync_interval must be positive")
        if self.learn_steps_per_iteration < 1:
            raise ValueError("learn_steps_per_iteration must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive when given")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("require 0 <= epsilon_end <= epsilon_start <= 1")
        if self.epsilon_decay_steps < 1:
            raise ValueError("epsilon_decay_steps must be positive")
        if self.noisy_sigma0 < 0.0:
            raise ValueError("noisy_sigma0 must be non-negative")
        if self.noisy_scope not in NOISY_SCOPES:
            raise ValueError(
                f"unknown noisy_scope {self.noisy_scope!r}; "
                f"expected one of {NOISY_SCOPES}"
            )
        if self.per_alpha < 0.0:
            raise ValueError("per_alpha must be non-negative")
        if not 0.0 <= self.per_beta_start <= self.per_beta_end <= 1.0:
            raise ValueError("require 0 <= per_beta_start <= per_beta_end <= 1")
        if self.per_beta_steps < 1:
            raise ValueError("per_beta_steps must be positive")
        if self.per_epsilon < 0.0:
            raise ValueError("per_epsilon must be non-negative")


@dataclass(frozen=True)
class BUIAConfig:
    """Bottom-up investor agent hyper-parameters (Algorithm 2 + Table 2).

    Published in Table 2: one convolutional layer with "kernel and filter counts
    matching the input", an MLP of 128 -> 128 -> 64 units, and a softmax output
    "over current candidate parcels".  Algorithm 2 adds: logits from a neural
    network, softmax, sample a categorical action, execute it, observe
    ``R_NN = Attract(s') - Attract(s)``.

    Everything below that carries a ``Design note`` on its own field.
    """

    observation_channels: int
    #: "within a 2 km radius" -> radius of two 1 km^2 parcels -> 5 x 5 window.
    observation_window: int = 5
    #: ``None`` reproduces Table 2 literally: filters = input channel count and
    #: kernel = the full observation window (so one parcel -> one feature
    #: vector).  Both stay overridable for ablations.
    conv_filters: int | None = None
    conv_kernel: int | None = None
    mlp_units: tuple[int, ...] = (128, 128, 64)
    #: Up to ten parcels may be converted per iteration (see EnvConfig).
    max_proposals: int = 10

    # --- training rule (not specified by the paper at all) ---
    #: Design note: REINFORCE (score-function gradient) on the categorical
    #: proposal policy with the paper's ``R_NN`` as the return.  Algorithm 2
    #: describes acting and observing a reward but never a learning rule;
    #: REINFORCE is the simplest estimator that matches "sample an action from
    #: Categorical(P)" without inventing a critic.
    learning_rate: float = 1e-3
    #: Design note: exponential moving-average reward baseline for variance
    #: reduction; ``1.0`` disables it (baseline stays at zero).
    baseline_decay: float = 0.99
    #: Design note: no entropy bonus by default (kept available for collapse).
    entropy_coefficient: float = 0.0
    max_grad_norm: float | None = 10.0
    #: Design note: ``R_NN = Attract(s') - Attract(s)`` needs a scalar summary of
    #: the attractiveness buffer.  "total" sums the change over the whole buffer
    #: B_attractiveness (the literal reading of a state-level quantity);
    #: "local" averages the change over the converted parcels' neighbourhoods.
    reward_mode: str = "total"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.observation_channels < 1:
            raise ValueError("observation_channels must be positive")
        if self.observation_window < 1 or self.observation_window % 2 == 0:
            raise ValueError("observation_window must be a positive odd number")
        if self.conv_filters is not None and self.conv_filters < 1:
            raise ValueError("conv_filters must be positive when given")
        if self.conv_kernel is not None and self.conv_kernel < 1:
            raise ValueError("conv_kernel must be positive when given")
        if not self.mlp_units:
            raise ValueError("at least one MLP layer is required")
        if any(units < 1 for units in self.mlp_units):
            raise ValueError("MLP layer widths must be positive")
        if self.max_proposals < 1:
            raise ValueError("max_proposals must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.baseline_decay <= 1.0:
            raise ValueError("baseline_decay must lie in [0, 1]")
        if self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive when given")
        if self.reward_mode not in ("total", "local"):
            raise ValueError("reward_mode must be 'total' or 'local'")
