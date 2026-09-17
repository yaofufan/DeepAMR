"""Residual network from O'Shea et al., IEEE JSTSP, 2018.

Over-the-Air Deep Learning Based Radio Signal Classification.
DOI: https://doi.org/10.1109/JSTSP.2018.2797022
"""

from __future__ import annotations

import math

import torch
from torch import nn


class OSheaResidualUnit(nn.Module):
    """Two linear convolutions with the skip connection shown in Figure 5."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x + residual


class OSheaResidualStack(nn.Module):
    """A 1x1 projection, residual units, and length-halving max pooling."""

    def __init__(
        self,
        in_channels: int,
        channels: int = 32,
        kernel_size: int = 3,
        units: int = 2,
    ) -> None:
        super().__init__()
        if units < 1:
            raise ValueError("Each residual stack must contain at least one unit.")
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.units = nn.Sequential(
            *(OSheaResidualUnit(channels, kernel_size) for _ in range(units))
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.units(self.projection(x)))


class OSheaResNet(nn.Module):
    """Narrow, deep 1D ResNet corresponding to Figure 5 and Table IV."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 2,
        input_length: int = 1024,
        hidden_channels: int = 32,
        dropout: float = 0.2,
        num_stacks: int = 6,
        units_per_stack: int = 2,
        kernel_size: int = 3,
        fc_channels: int = 128,
    ) -> None:
        super().__init__()
        downsample_factor = 2**num_stacks
        if input_length < downsample_factor:
            raise ValueError(
                f"input_length must be at least {downsample_factor} for "
                f"{num_stacks} residual stacks."
            )

        stacks = []
        channels = input_channels
        for _ in range(num_stacks):
            stacks.append(
                OSheaResidualStack(
                    channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    units=units_per_stack,
                )
            )
            channels = hidden_channels
        self.stacks = nn.Sequential(*stacks)

        pooled_length = input_length
        for _ in range(num_stacks):
            pooled_length //= 2
        self.feature_shape = (hidden_channels, pooled_length)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * pooled_length, fc_channels),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout),
            nn.Linear(fc_channels, fc_channels),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout),
            nn.Linear(fc_channels, num_classes),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(1.0 / module.in_features))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, channels, length], got {tuple(x.shape)}.")
        features = self.stacks(x)
        if tuple(features.shape[1:]) != self.feature_shape:
            raise ValueError(
                "Input shape does not match the configured input_length: "
                f"expected features {self.feature_shape}, got {tuple(features.shape[1:])}."
            )
        return self.classifier(features)
