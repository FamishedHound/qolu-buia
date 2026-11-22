"""QOLU network architecture (Table 1 of the paper).

    Conv. block    3x3 kernels, 32 -> 64 -> 128 filters
    Flatten layer
    Residual MLP   2 fully-connected layers (256 units) + skip connection
                   around the pair
    Quantile head  51 atoms with V in [-200, 200]

The head is dueling (Algorithm 1: "separate the value and advantage streams in
the network, then combine them"), so the 51 atoms are produced per action from
a mean-centred advantage stream plus a shared value stream.

Algorithm 1 line 5 also selects actions "by exploiting noisy network parameters
theta", so the fully-connected layers can be NoisyNet layers (Fortunato et al.
2018) - see :mod:`chapter4_qolu.agent.noisy` for the scope choice.  With
``noisy_nets=False`` the network is built out of plain ``nn.Linear`` layers in
a fixed construction order, so a seeded build is bit-identical from run to run.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chapter4_qolu.agent.distributional import AtomSupport, combine_dueling_streams
from chapter4_qolu.agent.noisy import (
    NOISY_SCOPES,
    linear_layer,
    reset_noise,
    zero_noise,
)

__all__ = ["QOLUNetwork", "QOLUNetworkConfig", "ResidualMLP"]


@dataclass(frozen=True)
class QOLUNetworkConfig:
    """Shape and size parameters of the QOLU network."""

    in_channels: int
    window: int
    n_actions: int
    conv_channels: tuple[int, ...] = (32, 64, 128)
    kernel_size: int = 3
    #: Design note: "same" padding, see chapter4_qolu.config.QOLUConfig.
    conv_padding: int = 1
    hidden_units: int = 256
    n_atoms: int = 51
    v_min: float = -200.0
    v_max: float = 200.0
    #: NoisyNet layers (Algorithm 1 line 5).  ``False`` builds the network out
    #: of plain ``nn.Linear`` layers.
    noisy_nets: bool = True
    noisy_sigma0: float = 0.5
    #: Which fully-connected layers carry noise; see
    #: :data:`chapter4_qolu.agent.noisy.NOISY_SCOPES`.
    noisy_scope: str = "all_linear"

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError("in_channels must be positive")
        if self.window < 1 or self.window % 2 == 0:
            raise ValueError("window must be a positive odd number")
        if self.n_actions < 1:
            raise ValueError("n_actions must be positive")
        if not self.conv_channels:
            raise ValueError("at least one convolution layer is required")
        if self.hidden_units < 1:
            raise ValueError("hidden_units must be positive")
        if self.n_atoms < 2:
            raise ValueError("n_atoms must be at least two")
        if self.v_max <= self.v_min:
            raise ValueError("v_max must be greater than v_min")
        if self.noisy_sigma0 < 0.0:
            raise ValueError("noisy_sigma0 must be non-negative")
        if self.noisy_scope not in NOISY_SCOPES:
            raise ValueError(
                f"unknown noisy_scope {self.noisy_scope!r}; expected one of "
                f"{NOISY_SCOPES}"
            )

    @property
    def trunk_is_noisy(self) -> bool:
        """Do the projection and the residual MLP carry noise as well?"""
        return self.noisy_nets and self.noisy_scope == "all_linear"


class ResidualMLP(nn.Module):
    """Two fully-connected layers with a skip connection around the pair."""

    def __init__(self, units: int, noisy: bool = False, sigma0: float = 0.5) -> None:
        super().__init__()
        self.block = nn.Sequential(
            linear_layer(units, units, noisy=noisy, sigma0=sigma0),
            nn.ReLU(inplace=True),
            linear_layer(units, units, noisy=noisy, sigma0=sigma0),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.relu(features + self.block(features))


class QOLUNetwork(nn.Module):
    """Categorical distributional dueling Q-network for a top-down agent."""

    def __init__(self, config: QOLUNetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.support = AtomSupport(
            n_atoms=config.n_atoms, v_min=config.v_min, v_max=config.v_max
        )

        layers: list[nn.Module] = []
        in_channels = config.in_channels
        for out_channels in config.conv_channels:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=config.kernel_size,
                    padding=config.conv_padding,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        flat_features = self._infer_flat_features()
        sigma0 = config.noisy_sigma0
        self.projection = nn.Sequential(
            linear_layer(
                flat_features,
                config.hidden_units,
                noisy=config.trunk_is_noisy,
                sigma0=sigma0,
            ),
            nn.ReLU(inplace=True),
        )
        self.residual_mlp = ResidualMLP(
            config.hidden_units, noisy=config.trunk_is_noisy, sigma0=sigma0
        )
        self.value_head = linear_layer(
            config.hidden_units, config.n_atoms, noisy=config.noisy_nets, sigma0=sigma0
        )
        self.advantage_head = linear_layer(
            config.hidden_units,
            config.n_actions * config.n_atoms,
            noisy=config.noisy_nets,
            sigma0=sigma0,
        )

    # ------------------------------------------------------------------
    def reset_noise(self) -> None:
        """Draw fresh NoisyNet noise; a no-op when ``noisy_nets`` is off."""
        reset_noise(self)

    def zero_noise(self) -> None:
        """Evaluate on the mean parameters ``mu``; a no-op without noise."""
        zero_noise(self)

    # ------------------------------------------------------------------
    def _infer_flat_features(self) -> int:
        with torch.no_grad():
            probe = torch.zeros(
                1, self.config.in_channels, self.config.window, self.config.window
            )
            return int(self.flatten(self.conv(probe)).shape[1])

    def features(self, observations: torch.Tensor) -> torch.Tensor:
        """Shared trunk: conv block -> flatten -> projection -> residual MLP."""
        if observations.ndim != 4:
            raise ValueError(
                "observations must have shape (batch, channels, window, window)"
            )
        hidden = self.projection(self.flatten(self.conv(observations)))
        return self.residual_mlp(hidden)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return per-action atom logits, shape ``(batch, n_actions, n_atoms)``."""
        hidden = self.features(observations)
        value = self.value_head(hidden).view(-1, 1, self.config.n_atoms)
        advantage = self.advantage_head(hidden).view(
            -1, self.config.n_actions, self.config.n_atoms
        )
        return combine_dueling_streams(value, advantage)

    def distribution(self, observations: torch.Tensor) -> torch.Tensor:
        """Return distributions ``Z(s, a)``, shape ``(batch, n_actions, n_atoms)``."""
        return torch.softmax(self(observations), dim=-1)

    def log_distribution(self, observations: torch.Tensor) -> torch.Tensor:
        """Numerically stable ``log Z(s, a)``."""
        return torch.log_softmax(self(observations), dim=-1)

    def q_values(self, observations: torch.Tensor) -> torch.Tensor:
        """Expected return per action, ``sum_i z_i p_i(s, a)``."""
        probabilities = self.distribution(observations)
        atoms = self.support.atoms(
            device=probabilities.device, dtype=probabilities.dtype
        )
        return (probabilities * atoms).sum(dim=-1)
