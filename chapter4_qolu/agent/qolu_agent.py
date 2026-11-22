"""Quantile-Optimized Land Use (QOLU) agent - Algorithm 1 of the paper.

Algorithm 1 combines four ingredients:

* a *categorical* return distribution on a fixed atom support (51 atoms over
  [-200, 200]) trained with a KL divergence, see
  :mod:`chapter4_qolu.agent.distributional` for the naming caveat;
* n-step returns ``R_t = sum_i gamma^i r_{t+i}`` (the bootstrap term is applied
  here rather than in the buffer);
* double Q-learning: "use argmax_a Q(s', a; theta) to select an action and
  theta^- to evaluate it";
* a dueling head: ``Q = V + (A - mean_a' A)``.

Reconciliation note: Algorithm 1 writes the bootstrap twice - once inside the
n-step return (line 12) and once in the double-Q line (line 14).  Applying both
would discount the next-state value twice, so this implementation uses the
standard, self-consistent reading: the target distribution is

    Phi( R_t^{(n)} + gamma^n * z ),   z ~ Z(s_{t+n}, argmax_a Q(s_{t+n}, a; theta); theta^-)

which is exactly the double-Q line with an n-step reward accumulation.

The single-actor loop implemented in :mod:`chapter4_qolu.runner` also stands in
for Algorithm 1's "for each actor k running in parallel": distributed actors
change throughput, not the learning rule.

Two further Rainbow components (Hessel et al. 2018) sit behind switches, because
QOLU combines the Rainbow components with a C51 head while the chapter states
only one of these two:

* ``noisy_nets`` (**on** by default) - Algorithm 1 line 5 selects an action "by
  exploiting noisy network parameters theta", so the NoisyNet layers are part
  of the published reading, with epsilon-greedy retained on top;
* ``prioritised_replay`` (**off** by default) - the chapter never mentions
  prioritisation, and Algorithm 1 asks only for "a minibatch of transitions
  from D", so uniform sampling stays the published reading and PER is an
  opt-in extra.

With both off the agent is Algorithm 1 with ordinary linear layers and a
uniform buffer; see :mod:`chapter4_qolu.agent.qolu_variants`.

Design note (what the return distribution does, and what it does not do).
The whole decision path reads ``Z(s, a)`` **only through its mean**:
:meth:`QOLUNetwork.q_values` collapses the 51 atoms to
``Q(s, a) = sum_i z_i p_i(s, a) = E[Z(s, a)]``; :meth:`QOLUAgent.act` takes
``argmax_a Q(s, a)``; :meth:`QOLUAgent.quality_scores` reports
``Q(s, approve) - Q(s, reject)``; and
:func:`chapter4_qolu.agent.consensus.consensus_decisions` sums those scores.
Two distributions with the same mean and different shape therefore produce the
same action, the same quality score and the same consensus - nothing
downstream of the mean is risk-sensitive, and no quantile, variance or tail of
``Z`` enters a decision anywhere in this package.

Where the distribution *does* act is the learning rule: the loss is the KL
divergence against the projected n-step target distribution, so the shape of
``Z`` shapes the gradients and therefore the mean that is learnt.  That makes
``Z`` an internal model rather than a decision variable.
Whether that internal model is any *good* is a measurable question, and
:mod:`chapter4_qolu.diagnostics` measures it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from chapter4_qolu.agent.checkpoint import QOLUCheckpointMixin
from chapter4_qolu.agent.distributional import (
    AtomSupport,
    categorical_kl_divergence,
    project_categorical_distribution,
)
from chapter4_qolu.agent.exploration import LinearEpsilonSchedule
from chapter4_qolu.agent.networks import QOLUNetwork, QOLUNetworkConfig
from chapter4_qolu.agent.prioritised_replay import (
    BetaSchedule,
    PrioritisedBatch,
    PrioritisedReplayBuffer,
)
from chapter4_qolu.agent.replay import NStepReplayBuffer, Transition, TransitionBatch
from chapter4_qolu.config import QOLUConfig

__all__ = ["QOLUAgent", "QOLUConfig"]

#: Index of the "approve this conversion" action in the QOLU action set.
_APPROVE = 1
#: Index of the "reject this conversion" action.
_REJECT = 0


class QOLUAgent(QOLUCheckpointMixin):
    """Top-down stakeholder agent appraising proposed land conversions."""

    def __init__(self, config: QOLUConfig, seed: int | None = None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        network_config = QOLUNetworkConfig(
            in_channels=config.observation_channels,
            window=config.observation_window,
            n_actions=config.n_actions,
            conv_channels=config.conv_channels,
            kernel_size=config.kernel_size,
            conv_padding=config.conv_padding,
            hidden_units=config.hidden_units,
            n_atoms=config.n_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            noisy_nets=config.noisy_nets,
            noisy_sigma0=config.noisy_sigma0,
            noisy_scope=config.noisy_scope,
        )
        self.online_network = QOLUNetwork(network_config).to(self.device)
        self.target_network = QOLUNetwork(network_config).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)
        self.target_network.eval()

        self.support = AtomSupport(
            n_atoms=config.n_atoms, v_min=config.v_min, v_max=config.v_max
        )
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
            eps=config.adam_eps,
        )
        self.beta_schedule = BetaSchedule(
            start=config.per_beta_start,
            end=config.per_beta_end,
            steps=config.per_beta_steps,
        )
        self.replay: NStepReplayBuffer = self._build_replay()
        self.epsilon_schedule = LinearEpsilonSchedule(
            start=config.epsilon_start,
            end=config.epsilon_end,
            decay_steps=config.epsilon_decay_steps,
        )
        self.steps = 0
        self.gradient_steps = 0
        #: How many times ``theta^- <- theta`` has actually fired.  Recorded
        #: because ``target_sync_interval`` is a free choice of the
        #: implementation (Algorithm 1 says "every ``t_n`` steps"; Table 1 never
        #: gives ``t_n``) and can be set longer than a whole run's gradient
        #: budget, in which case the reset never fires and the double-Q
        #: bootstrap stays on the randomly initialised target network from
        #: beginning to end.  The config says what was asked for; only this
        #: counter says what happened.
        self.target_syncs = 0
        #: The process-wide torch stream a checkpoint was written under, once
        #: one has been loaded; see :meth:`restore_torch_rng`.
        self.torch_rng_state: torch.Tensor | None = None

    def _build_replay(self) -> NStepReplayBuffer:
        """The buffer ``D`` - prioritised (Rainbow) or uniform (Algorithm 1)."""
        config = self.config
        if not config.prioritised_replay:
            return NStepReplayBuffer(
                capacity=config.replay_capacity,
                n_step=config.n_step,
                gamma=config.gamma,
            )
        return PrioritisedReplayBuffer(
            capacity=config.replay_capacity,
            n_step=config.n_step,
            gamma=config.gamma,
            alpha=config.per_alpha,
            epsilon=config.per_epsilon,
            beta=self.beta_schedule,
        )

    # ------------------------------------------------------------------
    # acting
    # ------------------------------------------------------------------
    def epsilon(self, step: int | None = None) -> float:
        """Exploration rate for ``step`` (defaults to the agent's own counter)."""
        return float(self.epsilon_schedule(self.steps if step is None else step))

    def beta(self, step: int | None = None) -> float:
        """Importance-sampling exponent for ``step`` gradient steps of PER."""
        return float(self.beta_schedule(self.gradient_steps if step is None else step))

    @torch.no_grad()
    def q_values(self, observations: np.ndarray) -> np.ndarray:
        """``Q(s, a) = E[Z(s, a)]`` for a batch, on the *mean* parameters.

        This is the **only** channel by which the return distribution reaches a
        decision: the 51 atoms are collapsed to their expectation here and
        nothing downstream ever sees the shape again (see the module docstring).

        Design note (noisy nets and the quality metric): the paper's "quality
        score" is a reported quantity that the cross-stakeholder consensus sums
        (Section 3.5), not a draw, so it is read off ``mu`` with the NoisyNet
        noise zeroed.  Only the action actually taken exploits the noisy
        parameters, which is what Algorithm 1 line 5 describes.
        """
        tensor = self._to_tensor(observations)
        was_training = self.online_network.training
        self.online_network.eval()
        self.online_network.zero_noise()
        values = self.online_network.q_values(tensor)
        self.online_network.train(was_training)
        return values.cpu().numpy()

    @torch.no_grad()
    def action_distributions(self, observations: np.ndarray) -> np.ndarray:
        """``p(s, a)`` over the atoms per action, on the *mean* parameters.

        Shape ``(batch, n_actions, n_atoms)``.  Nothing in the agent calls this:
        it exists so that :mod:`chapter4_qolu.diagnostics` can score the
        distribution the policy throws away, under exactly the parameter
        reading :meth:`q_values` uses.
        """
        tensor = self._to_tensor(observations)
        was_training = self.online_network.training
        self.online_network.eval()
        self.online_network.zero_noise()
        probabilities = self.online_network.distribution(tensor)
        self.online_network.train(was_training)
        return probabilities.cpu().numpy()

    @torch.no_grad()
    def noisy_q_values(self, observations: np.ndarray) -> np.ndarray:
        """``Q(s, a)`` under a *fresh* NoisyNet sample - Algorithm 1 line 5."""
        tensor = self._to_tensor(observations)
        was_training = self.online_network.training
        self.online_network.eval()
        self.online_network.reset_noise()
        values = self.online_network.q_values(tensor)
        self.online_network.train(was_training)
        return values.cpu().numpy()

    def quality_scores(self, observations: np.ndarray) -> np.ndarray:
        """The paper's "quality metric": how favourable a conversion is.

        Expressed as ``Q(s, approve) - Q(s, reject)`` so that positive means the
        stakeholder is in favour, which is what the cross-stakeholder consensus
        sum in :mod:`chapter4_qolu.agent.consensus` expects.  Both terms are
        means of ``Z``, so the quality metric - and therefore the consensus - is
        a function of ``E[Z]`` alone.
        """
        values = self.q_values(observations)
        return values[:, _APPROVE] - values[:, _REJECT]

    def act(self, observations: np.ndarray, epsilon: float | None = None) -> np.ndarray:
        """Appraise every candidate in the decision buffer - Algorithm 1 line 5.

        "Select action ``a`` by exploiting noisy network parameters ``theta``
        **or** exploratory action based on ``Epsilon``": the greedy action is
        taken under a freshly sampled NoisyNet draw when ``noisy_nets`` is on,
        and epsilon-greedy is applied on top of it either way (Design note: the
        line keeps both mechanisms, and Table 1 lists epsilon-greedy as *the*
        exploration row).
        """
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 4:
            raise ValueError(
                "observations must have shape (batch, channels, window, window)"
            )
        rate = self.epsilon() if epsilon is None else float(epsilon)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")

        values = (
            self.noisy_q_values(observations)
            if self.config.noisy_nets
            else self.q_values(observations)
        )
        actions = values.argmax(axis=1).astype(np.int64)
        if rate > 0.0:
            explore = self.rng.random(actions.shape[0]) < rate
            random_actions = self.rng.integers(
                0, self.config.n_actions, size=actions.shape[0]
            )
            actions = np.where(explore, random_actions, actions).astype(np.int64)
        self.steps += int(actions.shape[0])
        return actions

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def remember(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition ``(s, a, r, s')`` in the replay buffer ``D``."""
        self.replay.push(
            Transition(
                observation=np.asarray(observation, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_observation=np.asarray(next_observation, dtype=np.float32),
                done=bool(done),
            )
        )

    @torch.no_grad()
    def select_bootstrap_actions(self, next_observations: torch.Tensor) -> torch.Tensor:
        """Double Q-learning action selection with the *online* parameters."""
        return self.online_network.q_values(next_observations).argmax(dim=1)

    def _sample_batch(self) -> TransitionBatch:
        """One minibatch from ``D``, prioritised or uniform."""
        if isinstance(self.replay, PrioritisedReplayBuffer):
            return self.replay.sample(
                self.config.batch_size, self.rng, beta=self.beta()
            )
        return self.replay.sample(self.config.batch_size, self.rng)

    def learn(self) -> float | None:
        """One gradient step on ``D_KL(Z || Z(s, a; theta))``; ``None`` if not ready."""
        if len(self.replay) < max(self.config.learning_starts, self.config.batch_size):
            return None

        batch = self._sample_batch()
        observations = self._to_tensor(batch.observations)
        next_observations = self._to_tensor(batch.next_observations)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(
            batch.rewards, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        discounts = torch.as_tensor(
            batch.discounts, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            # Hessel et al. sample the NoisyNet noise independently for the
            # action-selection pass, the target network and the loss pass.
            self._resample_noise(self.online_network)
            bootstrap_actions = self.select_bootstrap_actions(next_observations)
            self._resample_noise(self.target_network)
            next_distribution = self.target_network.distribution(next_observations)
            next_probs = next_distribution[
                torch.arange(next_distribution.shape[0], device=self.device),
                bootstrap_actions,
            ]
            target_probs = project_categorical_distribution(
                next_probs=next_probs,
                rewards=rewards,
                dones=dones,
                discount=discounts,
                support=self.support,
            )

        self.online_network.train()
        self._resample_noise(self.online_network)
        log_probs = self.online_network.log_distribution(observations)
        chosen_log_probs = log_probs[
            torch.arange(log_probs.shape[0], device=self.device), actions
        ]
        loss = self._kl_loss(target_probs, chosen_log_probs, batch)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self.online_network.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_sync_interval == 0:
            self.sync_target()
        return float(loss.detach().cpu())

    def _kl_loss(
        self,
        target_probs: torch.Tensor,
        chosen_log_probs: torch.Tensor,
        batch: TransitionBatch,
    ) -> torch.Tensor:
        """``D_KL(Z || Z(s, a; theta))``, importance-weighted when PER is on.

        The per-sample divergence doubles as the transition's new priority
        (Schaul et al. use the TD error; the distributional loss is the KL, and
        Rainbow prioritises by it).
        """
        if not isinstance(batch, PrioritisedBatch) or not isinstance(
            self.replay, PrioritisedReplayBuffer
        ):
            return categorical_kl_divergence(target_probs, chosen_log_probs)

        per_sample = categorical_kl_divergence(
            target_probs, chosen_log_probs, reduction="none"
        )
        self.replay.update_priorities(
            batch.indices.tolist(),
            per_sample.detach().abs().cpu().numpy().tolist(),
        )
        weights = torch.as_tensor(
            batch.weights, dtype=per_sample.dtype, device=per_sample.device
        )
        return (per_sample * weights).mean()

    def _resample_noise(self, network: QOLUNetwork) -> None:
        """Fresh NoisyNet noise for one forward pass; a no-op without noise."""
        if self.config.noisy_nets:
            network.reset_noise()

    def sync_target(self) -> None:
        """``theta^- <- theta`` (Algorithm 1, last line)."""
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_syncs += 1

    # ------------------------------------------------------------------
    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(array, dtype=np.float32), dtype=torch.float32, device=self.device
        )
