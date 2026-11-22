"""BUIA network architecture (Table 2 of the paper).

    Conv. block   1 convolutional layer, kernel and filter counts matching the input
    MLP           128 -> 128 -> 64
    Output        Softmax over current candidate parcels

Design note: "kernel and filter counts matching the input" is read as *both*
sizes being fixed by the input tensor - the kernel spans the whole local
observation window (so a candidate parcel's 2 km neighbourhood collapses to one
feature vector) and the number of filters equals the number of input channels.

Design note: Table 2 stops at the 64-unit layer but the output has to be one
logit per candidate parcel before the softmax, so a final 64 -> 1 linear head is
added.  The softmax is then taken *across candidates*, not across units, which
is what "softmax over current candidate parcels" requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["BUIANetwork", "BUIANetworkConfig"]


@dataclass(frozen=True)
class BUIANetworkConfig:
    """Shape and size parameters of the BUIA proposal network."""

    in_channels: int
    window: int = 5
    conv_filters: int | None = None
    conv_kernel: int | None = None
    mlp_units: tuple[int, ...] = (128, 128, 64)

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError("in_channels must be positive")
        if self.window < 1 or self.window % 2 == 0:
            raise ValueError("window must be a positive odd number")
        if self.conv_filters is not None and self.conv_filters < 1:
            raise ValueError("conv_filters must be positive when given")
        if self.conv_kernel is not None and self.conv_kernel < 1:
            raise ValueError("conv_kernel must be positive when given")
        if self.conv_kernel is not None and self.conv_kernel > self.window:
            raise ValueError("conv_kernel cannot exceed the observation window")
        if not self.mlp_units:
            raise ValueError("at least one MLP layer is required")

    @property
    def filters(self) -> int:
        """Filter count, defaulting to "matching the input" channels."""
        return self.in_channels if self.conv_filters is None else self.conv_filters

    @property
    def kernel(self) -> int:
        """Kernel size, defaulting to "matching the input" window."""
        return self.window if self.conv_kernel is None else self.conv_kernel


class BUIANetwork(nn.Module):
    """Scores candidate parcels; a softmax over the scores gives the policy."""

    def __init__(self, config: BUIANetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.conv = nn.Sequential(
            nn.Conv2d(config.in_channels, config.filters, kernel_size=config.kernel),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()

        flat_features = self._infer_flat_features()
        layers: list[nn.Module] = []
        in_features = flat_features
        for units in config.mlp_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU(inplace=True))
            in_features = units
        self.mlp = nn.Sequential(*layers)
        self.score_head = nn.Linear(config.mlp_units[-1], 1)

    def _infer_flat_features(self) -> int:
        with torch.no_grad():
            probe = torch.zeros(
                1, self.config.in_channels, self.config.window, self.config.window
            )
            return int(self.flatten(self.conv(probe)).shape[1])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one proposal logit per candidate parcel, shape ``(n,)``."""
        if observations.ndim != 4:
            raise ValueError(
                "observations must have shape (candidates, channels, window, window)"
            )
        features = self.mlp(self.flatten(self.conv(observations)))
        return self.score_head(features).reshape(-1)

    def distribution(self, observations: torch.Tensor) -> torch.Tensor:
        """Softmax over the current candidate parcels."""
        return torch.softmax(self(observations), dim=0)

    def log_distribution(self, observations: torch.Tensor) -> torch.Tensor:
        """Numerically stable log of :meth:`distribution`."""
        return torch.log_softmax(self(observations), dim=0)
