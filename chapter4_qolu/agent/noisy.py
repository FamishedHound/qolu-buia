"""NoisyNet layers (Fortunato et al. 2018), the factorised Gaussian variant.

Algorithm 1 line 5 selects an action "by exploiting noisy network parameters
``theta`` **or** exploratory action based on ``Epsilon``", and QOLU combines
the Rainbow components (Hessel et al. 2018) with a C51 head.  Rainbow takes its
noisy layers from Fortunato et al., so that is what this module implements:

    y = (mu_w + sigma_w * eps_w) x + (mu_b + sigma_b * eps_b)

with *factorised* noise, ``eps_w[i, j] = f(eps_out[i]) f(eps_in[j])`` and
``eps_b[i] = f(eps_out[i])`` for ``f(x) = sgn(x) sqrt(|x|)``, which needs
``p + q`` unit Gaussians per layer instead of ``p * q``.

Initialisation follows Fortunato et al. section 3.1 for the factorised case:
``mu ~ U[-1/sqrt(p), 1/sqrt(p)]`` and ``sigma = sigma_0 / sqrt(p)``, with
``sigma_0 = 0.5`` (their Atari setting, and Rainbow's).

Noise is *explicit*, not tied to ``train()`` / ``eval()``
--------------------------------------------------------
The usual shortcut is to make ``forward`` noisy iff ``self.training``.  That
would silently disable NoisyNet here, because
:meth:`chapter4_qolu.agent.qolu_agent.QOLUAgent.q_values` evaluates the network
in ``eval()`` mode for *every* inference, acting included.  So instead the
layer always applies whatever is in its noise buffers, and the caller states
what it wants:

* :func:`reset_noise` - draw a fresh sample (before each training forward pass,
  and before each action selection);
* :func:`zero_noise` - fall back to the mean parameters ``mu``, which is what
  the deterministic *quality score* fed to the cross-stakeholder consensus is
  read off.

Both helpers are no-ops on a network built out of plain ``nn.Linear``, so the
``noisy_nets=False`` path in :func:`linear_layer` is unaffected by them.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from chapter4_qolu.config import NOISY_SCOPES

__all__ = [
    "NOISY_SCOPES",
    "NoisyLinear",
    "linear_layer",
    "reset_noise",
    "scaled_noise",
    "zero_noise",
]


def scaled_noise(values: torch.Tensor) -> torch.Tensor:
    """``f(x) = sgn(x) sqrt(|x|)``, the factorised-noise scaling function."""
    return values.sign() * values.abs().sqrt()


class NoisyLinear(nn.Module):
    """A linear layer whose weights carry learned factorised Gaussian noise."""

    def __init__(
        self, in_features: int, out_features: int, sigma0: float = 0.5
    ) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError("in_features must be positive")
        if out_features < 1:
            raise ValueError("out_features must be positive")
        if sigma0 < 0.0:
            raise ValueError("sigma0 must be non-negative")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma0 = float(sigma0)

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    # ------------------------------------------------------------------
    def reset_parameters(self) -> None:
        """Fortunato et al. section 3.1, factorised case."""
        bound = 1.0 / math.sqrt(self.in_features)
        sigma = self.sigma0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.weight_mu.uniform_(-bound, bound)
            self.bias_mu.uniform_(-bound, bound)
            self.weight_sigma.fill_(sigma)
            self.bias_sigma.fill_(sigma)

    @torch.no_grad()
    def reset_noise(self) -> None:
        """Draw ``p + q`` unit Gaussians and form the rank-one noise matrix."""
        device = self.weight_mu.device
        dtype = self.weight_mu.dtype
        epsilon_in = scaled_noise(
            torch.randn(self.in_features, device=device, dtype=dtype)
        )
        epsilon_out = scaled_noise(
            torch.randn(self.out_features, device=device, dtype=dtype)
        )
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def zero_noise(self) -> None:
        """Fall back to the mean parameters ``mu`` until the next resample."""
        self.weight_epsilon.zero_()
        self.bias_epsilon.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return nn.functional.linear(features, weight, bias)

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"sigma0={self.sigma0}"
        )


def reset_noise(module: nn.Module) -> None:
    """Resample the noise of every :class:`NoisyLinear` inside ``module``."""
    for child in module.modules():
        if isinstance(child, NoisyLinear):
            child.reset_noise()


def zero_noise(module: nn.Module) -> None:
    """Switch every :class:`NoisyLinear` inside ``module`` to its mean weights."""
    for child in module.modules():
        if isinstance(child, NoisyLinear):
            child.zero_noise()


def linear_layer(
    in_features: int, out_features: int, noisy: bool, sigma0: float = 0.5
) -> nn.Module:
    """A fully-connected layer, noisy or not.

    The ``noisy=False`` branch returns a bare ``nn.Linear`` with its default
    initialisation, so a seeded build consumes exactly the random numbers
    ``nn.Linear`` does and produces the same parameters every time.
    """
    if not noisy:
        return nn.Linear(in_features, out_features)
    return NoisyLinear(in_features, out_features, sigma0=sigma0)
