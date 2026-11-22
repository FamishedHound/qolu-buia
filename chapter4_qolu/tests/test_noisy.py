"""Tests for the NoisyNet layers (Fortunato et al. 2018, factorised Gaussian)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from chapter4_qolu.agent.noisy import (
    NoisyLinear,
    linear_layer,
    reset_noise,
    scaled_noise,
    zero_noise,
)


class TestParameterInitialisation:
    def test_exposes_mu_and_sigma_for_weight_and_bias(self) -> None:
        layer = NoisyLinear(8, 4)
        names = {name for name, _ in layer.named_parameters()}
        assert names == {"weight_mu", "weight_sigma", "bias_mu", "bias_sigma"}
        assert layer.weight_mu.shape == (4, 8)
        assert layer.weight_sigma.shape == (4, 8)
        assert layer.bias_mu.shape == (4,)
        assert layer.bias_sigma.shape == (4,)

    def test_noise_is_a_buffer_not_a_parameter(self) -> None:
        layer = NoisyLinear(8, 4)
        buffers = {name for name, _ in layer.named_buffers()}
        assert buffers == {"weight_epsilon", "bias_epsilon"}

    def test_mu_is_uniform_on_plus_minus_one_over_root_p(self) -> None:
        torch.manual_seed(0)
        layer = NoisyLinear(400, 200)
        bound = 1.0 / math.sqrt(400)
        assert float(layer.weight_mu.detach().min()) >= -bound
        assert float(layer.weight_mu.detach().max()) <= bound
        # A uniform sample on a symmetric interval has mean 0 and sd bound/sqrt(3).
        assert float(layer.weight_mu.detach().mean()) == pytest.approx(
            0.0, abs=bound / 20
        )
        assert float(layer.weight_mu.detach().std()) == pytest.approx(
            bound / math.sqrt(3), rel=0.05
        )
        assert float(layer.bias_mu.detach().min()) >= -bound
        assert float(layer.bias_mu.detach().max()) <= bound

    def test_sigma_is_sigma0_over_root_p(self) -> None:
        layer = NoisyLinear(64, 8, sigma0=0.5)
        expected = 0.5 / math.sqrt(64)
        assert torch.allclose(layer.weight_sigma, torch.full((8, 64), expected))
        assert torch.allclose(layer.bias_sigma, torch.full((8,), expected))

    def test_sigma0_is_configurable(self) -> None:
        layer = NoisyLinear(16, 4, sigma0=0.1)
        assert float(layer.weight_sigma.detach()[0, 0]) == pytest.approx(0.1 / 4.0)

    def test_rejects_invalid_shapes(self) -> None:
        with pytest.raises(ValueError):
            NoisyLinear(0, 4)
        with pytest.raises(ValueError):
            NoisyLinear(4, 0)
        with pytest.raises(ValueError):
            NoisyLinear(4, 4, sigma0=-1.0)


class TestFactorisedNoise:
    def test_scaled_noise_is_signed_square_root(self) -> None:
        values = torch.tensor([-4.0, -1.0, 0.0, 1.0, 9.0])
        torch.testing.assert_close(
            scaled_noise(values), torch.tensor([-2.0, -1.0, 0.0, 1.0, 3.0])
        )

    def test_weight_noise_is_the_rank_one_outer_product(self) -> None:
        """Factorised noise: eps_w[i, j] = f(eps_out[i]) f(eps_in[j])."""
        torch.manual_seed(3)
        layer = NoisyLinear(6, 5)
        layer.reset_noise()
        matrix = layer.weight_epsilon
        # A rank-one matrix has exactly one non-zero singular value.
        singular = torch.linalg.svdvals(matrix)
        assert float(singular[0]) > 1e-6
        assert float(singular[1:].max()) < 1e-4 * float(singular[0])

    def test_bias_noise_is_the_output_factor(self) -> None:
        """eps_b = f(eps_out), i.e. the column factor of the weight noise."""
        torch.manual_seed(4)
        layer = NoisyLinear(6, 5)
        layer.reset_noise()
        # weight_epsilon[:, j] is eps_b scaled by the scalar f(eps_in[j]).
        column = layer.weight_epsilon[:, 0]
        scale = column / layer.bias_epsilon
        assert float(scale.std()) < 1e-5

    def test_reset_noise_draws_a_fresh_sample(self) -> None:
        torch.manual_seed(5)
        layer = NoisyLinear(6, 5)
        first = layer.weight_epsilon.clone()
        layer.reset_noise()
        assert not torch.allclose(first, layer.weight_epsilon)

    def test_noise_is_centred_across_many_draws(self) -> None:
        torch.manual_seed(6)
        layer = NoisyLinear(32, 32)
        draws = []
        for _ in range(200):
            layer.reset_noise()
            draws.append(layer.bias_epsilon.clone())
        stacked = torch.stack(draws)
        assert float(stacked.mean()) == pytest.approx(0.0, abs=0.05)
        # f(x) = sgn(x) sqrt(|x|) of a standard normal has variance E|x| = sqrt(2/pi).
        assert float(stacked.var()) == pytest.approx(math.sqrt(2.0 / math.pi), rel=0.1)

    def test_zero_noise_restores_the_mean_parameters(self) -> None:
        torch.manual_seed(7)
        layer = NoisyLinear(6, 5)
        layer.zero_noise()
        assert float(layer.weight_epsilon.abs().max()) == 0.0
        assert float(layer.bias_epsilon.abs().max()) == 0.0


class TestForward:
    def test_output_shape(self) -> None:
        layer = NoisyLinear(6, 5)
        assert layer(torch.randn(3, 6)).shape == (3, 5)

    def test_noise_free_forward_equals_the_mean_linear_map(self) -> None:
        torch.manual_seed(8)
        layer = NoisyLinear(6, 5)
        layer.zero_noise()
        features = torch.randn(4, 6)
        expected = torch.nn.functional.linear(features, layer.weight_mu, layer.bias_mu)
        torch.testing.assert_close(layer(features), expected)

    def test_a_noisy_forward_differs_from_the_mean_map(self) -> None:
        torch.manual_seed(9)
        layer = NoisyLinear(6, 5)
        layer.reset_noise()
        features = torch.randn(4, 6)
        mean_map = torch.nn.functional.linear(features, layer.weight_mu, layer.bias_mu)
        assert not torch.allclose(layer(features), mean_map)

    def test_noise_is_held_fixed_until_it_is_resampled(self) -> None:
        torch.manual_seed(10)
        layer = NoisyLinear(6, 5)
        features = torch.randn(4, 6)
        layer.reset_noise()
        first = layer(features)
        torch.testing.assert_close(layer(features), first)
        layer.reset_noise()
        assert not torch.allclose(layer(features), first)

    def test_gradients_reach_both_mu_and_sigma(self) -> None:
        torch.manual_seed(11)
        layer = NoisyLinear(6, 5)
        layer.reset_noise()
        layer(torch.randn(4, 6)).square().sum().backward()
        for name, parameter in layer.named_parameters():
            assert parameter.grad is not None, name
            assert float(parameter.grad.abs().sum()) > 0.0, name

    def test_sigma_zero_makes_the_layer_deterministic(self) -> None:
        torch.manual_seed(12)
        layer = NoisyLinear(6, 5, sigma0=0.0)
        features = torch.randn(4, 6)
        layer.reset_noise()
        first = layer(features)
        layer.reset_noise()
        torch.testing.assert_close(layer(features), first)


class TestModuleHelpers:
    def test_reset_and_zero_noise_recurse_into_submodules(self) -> None:
        torch.manual_seed(13)
        module = nn.Sequential(NoisyLinear(4, 4), nn.ReLU(), NoisyLinear(4, 2))
        reset_noise(module)
        assert all(
            float(child.weight_epsilon.abs().sum()) > 0.0
            for child in module
            if isinstance(child, NoisyLinear)
        )
        zero_noise(module)
        assert all(
            float(child.weight_epsilon.abs().max()) == 0.0
            for child in module
            if isinstance(child, NoisyLinear)
        )

    def test_helpers_are_a_no_op_on_a_plain_network(self) -> None:
        module = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        before = [p.detach().clone() for p in module.parameters()]
        reset_noise(module)
        zero_noise(module)
        for parameter, original in zip(module.parameters(), before):
            torch.testing.assert_close(parameter, original)


class TestLinearFactory:
    """The factory is what keeps ``noisy_nets=False`` on the plain ``nn.Linear`` path."""

    def test_returns_a_plain_linear_when_noise_is_off(self) -> None:
        layer = linear_layer(6, 5, noisy=False, sigma0=0.5)
        assert type(layer) is nn.Linear

    def test_returns_a_noisy_linear_when_noise_is_on(self) -> None:
        layer = linear_layer(6, 5, noisy=True, sigma0=0.5)
        assert isinstance(layer, NoisyLinear)
        assert float(layer.weight_sigma.detach()[0, 0]) == pytest.approx(
            0.5 / math.sqrt(6)
        )

    def test_the_plain_path_consumes_the_same_random_numbers_as_nn_linear(self) -> None:
        torch.manual_seed(14)
        made = linear_layer(6, 5, noisy=False, sigma0=0.5)
        torch.manual_seed(14)
        reference = nn.Linear(6, 5)
        torch.testing.assert_close(made.weight, reference.weight)
        torch.testing.assert_close(made.bias, reference.bias)
