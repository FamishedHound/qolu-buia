"""One P-MADDPG agent: actor ``upsilon_i``, critic ``psi_i`` and their targets.

Split out of :mod:`chapter4_qolu.baselines.pmaddpg` to keep both files small;
the algorithmic content is Chapter 3's Algorithm 1, lines 3-12.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from chapter4_qolu.baselines.config import PMADDPGConfig
from chapter4_qolu.baselines.pmaddpg_networks import CentralisedCritic, ParcelActor

__all__ = ["PMADDPGAgent", "target_q_value"]


@contextmanager
def _local_torch_seed(seed: int | None) -> Iterator[None]:
    """Seed torch for the duration of the block, leaving the global RNG alone.

    ``torch.manual_seed`` mutates the *process-wide* generator, so seeding each
    agent that way would make one method's construction reorder every other
    torch draw in the process - including another method's.  ``fork_rng`` saves and
    restores the CPU generator around the block instead; ``devices=[]`` keeps it
    off the CUDA path, which this CPU-only baseline never uses.
    """
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


def target_q_value(
    next_q: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    dones: torch.Tensor | None = None,
    mask_terminal: bool = False,
) -> torch.Tensor:
    """``targetQ <- psi_i^t(s', upsilon_i^t(s')) * gamma + r``.

    Written exactly as Chapter 3's Algorithm 1 line 6, i.e. **without** a
    ``(1 - done)`` factor.  Design note: that omission is followed literally here
    because it is what the published pseudo-code says; ``mask_terminal=True``
    restores the standard masked form for anyone who prefers it.
    """
    values = next_q.reshape(-1, 1)
    reward = rewards.reshape(-1, 1)
    if mask_terminal and dones is not None:
        values = values * (1.0 - dones.reshape(-1, 1))
    return values * float(gamma) + reward


class PMADDPGAgent:
    """One stakeholder: actor ``upsilon_i``, critic ``psi_i`` and their targets."""

    def __init__(self, config: PMADDPGConfig, index: int, seed: int | None) -> None:
        self.config = config
        self.index = int(index)
        self.device = torch.device(config.device)
        #: Learning updates performed, i.e. Chapter 3's "steps" for the target
        #: refresh of :meth:`update_targets`.
        self.updates = 0
        with _local_torch_seed(seed):
            self._build(config)

    def _build(self, config: PMADDPGConfig) -> None:
        """Create both networks, both targets and both optimisers."""

        def build_actor() -> ParcelActor:
            return ParcelActor(
                in_channels=config.observation_channels,
                window=config.observation_window,
                conv_channels=config.conv_channels,
                kernel_size=config.kernel_size,
                padding=config.conv_padding,
                hidden_units=config.hidden_units,
            ).to(self.device)

        def build_critic() -> CentralisedCritic:
            return CentralisedCritic(
                in_channels=config.observation_channels,
                window=config.observation_window,
                n_agents=config.n_agents,
                conv_channels=config.conv_channels,
                kernel_size=config.kernel_size,
                padding=config.conv_padding,
                hidden_units=config.hidden_units,
            ).to(self.device)

        self.actor = build_actor()
        self.actor_target = build_actor()
        self.critic = build_critic()
        self.critic_target = build_critic()
        self._freeze(self.actor_target, self.actor)
        self._freeze(self.critic_target, self.critic)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )

    @staticmethod
    def _freeze(target: nn.Module, source: nn.Module) -> None:
        target.load_state_dict(source.state_dict())
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        target.eval()

    # ------------------------------------------------------------------
    # acting
    # ------------------------------------------------------------------
    @torch.no_grad()
    def approve_scores(self, observations: torch.Tensor) -> np.ndarray:
        was_training = self.actor.training
        self.actor.eval()
        scores = self.actor(observations).reshape(-1)
        self.actor.train(was_training)
        return scores.cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def update_critic(
        self,
        observations: torch.Tensor,
        joint_actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_joint_actions: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """Chapter 3, Algorithm 1 lines 3-8 (MSE against the published target)."""
        with torch.no_grad():
            next_q = self.critic_target(next_observations, next_joint_actions)
            target = target_q_value(
                next_q,
                rewards,
                self.config.gamma,
                dones=dones,
                mask_terminal=self.config.mask_terminal,
            )
        current = self.critic(observations, joint_actions)
        loss = torch.nn.functional.mse_loss(current, target)

        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._clip(self.critic)
        self.critic_optimizer.step()
        return float(loss.detach().cpu())

    def policy_value(
        self, observations: torch.Tensor, joint_actions: torch.Tensor, index: int
    ) -> float:
        """``mean psi_i(s, upsilon_i(s))`` - the quantity the actor maximises."""
        with torch.no_grad():
            joint = self._substitute(joint_actions, self.actor(observations), index)
            return float(self.critic(observations, joint).mean().cpu())

    def update_actor(
        self, observations: torch.Tensor, joint_actions: torch.Tensor, index: int
    ) -> float:
        """Chapter 3, Algorithm 1 lines 10-12: ``loss = -mean psi_i(s, upsilon_i(s))``.

        Design note: Chapter 3 writes only agent ``i``'s action in the critic
        call.  With a centralised critic the other agents' actions still have to
        come from somewhere, so - following standard MADDPG - agent ``i``'s slot
        is replaced by its current actor output and the remaining slots are
        taken from the mini-batch.

        Design note (``config.policy_regulariser``, default 1e-3): the published
        loss is the first term alone.  Chapter 3's action is a *continuous*
        cell parameter, whereas this environment's is binary, so this
        implementation squashes the actor through a sigmoid - and a squashed
        actor under the bare loss is a trap.  The pre-activation is unbounded
        and the loss rewards pushing it outwards, so within a handful of
        updates ``|logit|`` reaches 37-92, ``sigma'(z)`` underflows, and no
        later critic gradient can move the policy again: the run locks into a
        random approve-all / reject-all corner.  Lowe et al.'s own MADDPG
        implementation adds ``1e-3 * mean(logits ** 2)`` to every actor loss
        precisely to keep the pre-activation in range, so that term is used
        here.  Setting the
        coefficient to 0.0 restores the literal published loss.
        """
        logits = self.actor.logits(observations)
        joint = self._substitute(joint_actions, torch.sigmoid(logits), index)
        loss = -self.critic(observations, joint).mean()
        if self.config.policy_regulariser > 0.0:
            loss = loss + self.config.policy_regulariser * (logits**2).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._clip(self.actor)
        self.actor_optimizer.step()
        self.updates += 1
        return float(loss.detach().cpu())

    @staticmethod
    def _substitute(
        joint_actions: torch.Tensor, own_action: torch.Tensor, index: int
    ) -> torch.Tensor:
        columns = list(torch.unbind(joint_actions, dim=1))
        columns[index] = own_action.reshape(-1)
        return torch.stack(columns, dim=1)

    def _clip(self, module: nn.Module) -> None:
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(module.parameters(), self.config.max_grad_norm)

    @property
    def _target_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        return (
            (self.actor_target, self.actor),
            (self.critic_target, self.critic),
        )

    def update_targets(self) -> None:
        """Refresh both target networks the way Chapter 3 says.

        PUBLISHED (Chapter 3 line 392): "We update the networks to the target
        networks every 250 steps" - a hard copy on a fixed interval, which is
        what ``config.target_update_interval`` counts (in learning updates of
        this agent).  ``None`` selects Polyak averaging at ``config.tau``
        instead, which is the thesis's own Appendix B reading ("Target network
        update rate tau = 0.01") and what ``variant="v1"`` keeps.
        """
        interval = self.config.target_update_interval
        if interval is None:
            self.soft_update()
        elif self.updates % int(interval) == 0:
            self.hard_update()

    def soft_update(self) -> None:
        """Polyak averaging of both target networks."""
        tau = self.config.tau
        for target, source in self._target_pairs:
            with torch.no_grad():
                for target_parameter, parameter in zip(
                    target.parameters(), source.parameters()
                ):
                    target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)

    def hard_update(self) -> None:
        """Copy both online networks onto their targets verbatim."""
        for target, source in self._target_pairs:
            target.load_state_dict(source.state_dict())
