"""``noisy_nets=False`` / ``prioritised_replay=False`` *are* the plain C51 agent.

The QOLU agent has two Rainbow switches.  Neither may change anything when it
is off, because ``qolu_c51`` has to stay a faithful reference variant.  "Off is
a no-op" is therefore tested rather than asserted: the plain layer path is
built here independently and compared parameter by parameter, forward pass
and backward pass, and the agent's numpy stream is checked to advance by
exactly one uniform minibatch draw per learning step - no extra random numbers
are consumed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from chapter4_qolu.agent.networks import QOLUNetwork, QOLUNetworkConfig, ResidualMLP
from chapter4_qolu.agent.noisy import NoisyLinear
from chapter4_qolu.agent.replay import NStepReplayBuffer
from chapter4_qolu.tests.test_qolu_rainbow import (
    C51_ONLY,
    fill,
    make_agent,
    observations,
)


def reference_pre_rainbow_modules(
    in_channels: int, window: int, n_actions: int, hidden: int = 256, atoms: int = 51
) -> list[nn.Module]:
    """The plain layer path, built independently in the agent's construction order.

    This is what makes "``noisy_nets=False`` builds the plain network exactly"
    testable: the convolutions, the projection, the residual MLP and the two
    heads are created out of ``nn.Conv2d`` / ``nn.Linear`` in exactly this
    order, so a seeded build consumes the same random numbers and must produce
    the same parameters.
    """
    modules: list[nn.Module] = []
    channels = in_channels
    for out_channels in (32, 64, 128):
        modules.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1))
        channels = out_channels
    flat = channels * window * window
    modules.append(nn.Linear(flat, hidden))
    modules.append(ResidualMLP(hidden))
    modules.append(nn.Linear(hidden, atoms))
    modules.append(nn.Linear(hidden, n_actions * atoms))
    return modules


class TestNoisyOffIsThePreRainbowNetwork:
    def test_every_layer_is_a_plain_linear(self) -> None:
        net = QOLUNetwork(
            QOLUNetworkConfig(in_channels=4, window=5, n_actions=2, noisy_nets=False)
        )
        assert [m for m in net.modules() if isinstance(m, NoisyLinear)] == []
        assert type(net.value_head) is nn.Linear
        assert type(net.advantage_head) is nn.Linear

    def test_seeded_parameters_match_the_pre_rainbow_layer_path(self) -> None:
        torch.manual_seed(20260924)
        net = QOLUNetwork(
            QOLUNetworkConfig(in_channels=4, window=5, n_actions=2, noisy_nets=False)
        )
        torch.manual_seed(20260924)
        reference = reference_pre_rainbow_modules(4, 5, 2)
        expected = [p.detach() for module in reference for p in module.parameters()]
        actual = [p.detach() for p in net.parameters()]
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            assert torch.equal(got, want)

    def test_seeded_forward_and_backward_match_the_pre_rainbow_layer_path(
        self,
    ) -> None:
        torch.manual_seed(7)
        net = QOLUNetwork(
            QOLUNetworkConfig(in_channels=4, window=5, n_actions=2, noisy_nets=False)
        )
        torch.manual_seed(7)
        reference = reference_pre_rainbow_modules(4, 5, 2)
        conv, projection, residual, value, advantage = (
            nn.Sequential(
                reference[0],
                nn.ReLU(),
                reference[1],
                nn.ReLU(),
                reference[2],
                nn.ReLU(),
            ),
            reference[3],
            reference[4],
            reference[5],
            reference[6],
        )

        torch.manual_seed(99)
        batch = torch.randn(3, 4, 5, 5)
        hidden = residual(torch.relu(projection(torch.flatten(conv(batch), 1))))
        value_logits = value(hidden).view(-1, 1, 51)
        advantage_logits = advantage(hidden).view(-1, 2, 51)
        expected = (
            value_logits + advantage_logits - advantage_logits.mean(dim=1, keepdim=True)
        )
        produced = net(batch)
        assert torch.equal(produced, expected)

        produced.square().sum().backward()
        expected.square().sum().backward()
        for got, want in zip(
            net.parameters(), (p for module in reference for p in module.parameters())
        ):
            assert torch.equal(got.grad, want.grad)

    def test_noise_helpers_are_a_no_op_when_noise_is_off(self) -> None:
        torch.manual_seed(8)
        net = QOLUNetwork(
            QOLUNetworkConfig(in_channels=4, window=5, n_actions=2, noisy_nets=False)
        )
        batch = torch.randn(2, 4, 5, 5)
        before = net(batch)
        net.reset_noise()
        assert torch.equal(net(batch), before)
        net.zero_noise()
        assert torch.equal(net(batch), before)


class TestBothSwitchesOffIsThePreRainbowAgent:
    def test_the_replay_buffer_is_the_uniform_one(self) -> None:
        agent = make_agent(**C51_ONLY)
        assert type(agent.replay) is NStepReplayBuffer

    def test_the_networks_have_no_noisy_layers(self) -> None:
        agent = make_agent(**C51_ONLY)
        for network in (agent.online_network, agent.target_network):
            assert [m for m in network.modules() if isinstance(m, NoisyLinear)] == []

    def test_learning_draws_exactly_one_uniform_minibatch_per_step(self) -> None:
        """No extra random draws: the numpy stream advances by one minibatch draw."""
        agent = make_agent(**C51_ONLY)
        fill(agent)
        reference = np.random.default_rng(0)
        # Replay the draws ``act`` never made, then the one ``learn`` makes.
        state_before = agent.rng.bit_generator.state
        reference.bit_generator.state = state_before
        agent.learn()
        reference.integers(0, len(agent.replay), size=agent.config.batch_size)
        assert agent.rng.bit_generator.state == reference.bit_generator.state

    def test_two_seeded_agents_learn_identically(self) -> None:
        losses = []
        for _ in range(2):
            agent = make_agent(**C51_ONLY)
            fill(agent)
            losses.append([agent.learn() for _ in range(5)])
        assert losses[0] == losses[1]

    def test_acting_uses_the_deterministic_q_values(self) -> None:
        agent = make_agent(**C51_ONLY)
        obs = observations(5)
        np.testing.assert_array_equal(
            agent.act(obs, epsilon=0.0), agent.q_values(obs).argmax(axis=1)
        )
