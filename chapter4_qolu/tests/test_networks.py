"""Tests for the QOLU network (Table 1 of the paper)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from chapter4_qolu.agent.networks import QOLUNetwork, QOLUNetworkConfig
from chapter4_qolu.agent.noisy import NoisyLinear

#: Table 1's "fully-connected layer", whichever reading of Algorithm 1 line 5
#: is in force: a plain ``nn.Linear`` or its NoisyNet equivalent.  Both expose
#: ``in_features`` / ``out_features``, so the shape assertions below are the
#: same either way.
FULLY_CONNECTED = (nn.Linear, NoisyLinear)


def zero_fully_connected(layer: nn.Module) -> None:
    """Turn one fully-connected layer into the zero map, noisy or not."""
    with torch.no_grad():
        if isinstance(layer, NoisyLinear):
            layer.weight_mu.zero_()
            layer.weight_sigma.zero_()
            layer.bias_mu.zero_()
            layer.bias_sigma.zero_()
        else:
            layer.weight.zero_()
            layer.bias.zero_()


@pytest.fixture()
def network() -> QOLUNetwork:
    torch.manual_seed(0)
    config = QOLUNetworkConfig(in_channels=11, window=9, n_actions=2)
    return QOLUNetwork(config)


class TestArchitectureMatchesTable1:
    def test_default_hyperparameters(self) -> None:
        config = QOLUNetworkConfig(in_channels=11, window=9, n_actions=2)
        assert config.conv_channels == (32, 64, 128)
        assert config.kernel_size == 3
        assert config.hidden_units == 256
        assert config.n_atoms == 51
        assert config.v_min == pytest.approx(-200.0)
        assert config.v_max == pytest.approx(200.0)

    def test_convolution_block_filters(self, network: QOLUNetwork) -> None:
        convs = [m for m in network.modules() if isinstance(m, nn.Conv2d)]
        assert [c.out_channels for c in convs] == [32, 64, 128]
        assert all(c.kernel_size == (3, 3) for c in convs)
        assert convs[0].in_channels == 11

    def test_residual_mlp_has_two_256_unit_layers(self, network: QOLUNetwork) -> None:
        widths = [
            (m.in_features, m.out_features)
            for m in network.residual_mlp.modules()
            if isinstance(m, FULLY_CONNECTED)
        ]
        assert widths.count((256, 256)) == 2

    def test_residual_mlp_applies_a_skip_connection(self, network: QOLUNetwork) -> None:
        features = torch.randn(4, 256)
        with torch.no_grad():
            for module in network.residual_mlp.block.modules():
                if isinstance(module, FULLY_CONNECTED):
                    zero_fully_connected(module)
            out = network.residual_mlp(features)
        # With a zeroed inner block the residual path must pass the input through.
        torch.testing.assert_close(out, torch.relu(features))

    def test_atom_head_outputs_51_atoms_per_action(self, network: QOLUNetwork) -> None:
        logits = network(torch.zeros(3, 11, 9, 9))
        assert logits.shape == (3, 2, 51)


class TestForwardSemantics:
    def test_distribution_is_normalised_over_atoms(self, network: QOLUNetwork) -> None:
        probs = network.distribution(torch.randn(5, 11, 9, 9))
        assert probs.shape == (5, 2, 51)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(5, 2))
        assert torch.all(probs >= 0)

    def test_q_values_are_the_expectation_over_the_atom_support(
        self, network: QOLUNetwork
    ) -> None:
        observations = torch.randn(5, 11, 9, 9)
        probs = network.distribution(observations)
        atoms = network.support.atoms()
        expected = (probs * atoms).sum(dim=-1)
        torch.testing.assert_close(network.q_values(observations), expected)

    def test_identical_advantages_give_identical_action_values(self) -> None:
        torch.manual_seed(1)
        config = QOLUNetworkConfig(in_channels=3, window=7, n_actions=4)
        net = QOLUNetwork(config)
        zero_fully_connected(net.advantage_head)
        q_values = net.q_values(torch.randn(2, 3, 7, 7))
        spread = q_values.max(dim=1).values - q_values.min(dim=1).values
        assert torch.all(spread < 1e-5)

    def test_gradients_reach_every_parameter(self, network: QOLUNetwork) -> None:
        logits = network(torch.randn(2, 11, 9, 9))
        logits.sum().backward()
        missing = [name for name, p in network.named_parameters() if p.grad is None]
        assert missing == []

    def test_handles_various_window_sizes(self) -> None:
        for window in (5, 7, 9, 11):
            net = QOLUNetwork(
                QOLUNetworkConfig(in_channels=4, window=window, n_actions=2)
            )
            assert net(torch.zeros(1, 4, window, window)).shape == (1, 2, 51)

    def test_rejects_observations_with_the_wrong_channel_count(
        self, network: QOLUNetwork
    ) -> None:
        with pytest.raises((RuntimeError, ValueError)):
            network(torch.zeros(1, 5, 9, 9))
