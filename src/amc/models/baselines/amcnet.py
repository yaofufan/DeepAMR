"""PyTorch reproduction of AMC-Net for automatic modulation recognition.

Reference:
    Zhang et al., "AMC-Net: An Effective Network for Automatic
    Modulation Classification," ICASSP, 2023.
    Paper: https://arxiv.org/abs/2304.00445
    Code: https://github.com/zjwfufu/AMC-Net (models/model.py)

Implementation:
    Group-member port of the official PyTorch implementation.

Input: I/Q tensors with shape [N, 2, L].
Output: Unnormalized class logits with shape [N, num_classes].
Note: Frequency correction uses float32 FFT under mixed precision.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """(1 x 3) convolution, ReLU, BatchNorm (official Conv_Block)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleModule(nn.Module):
    """MSM: parallel (2x3), (2x5), (2x7) convolutions across the I/Q rows."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        if out_channels % 3 != 0:
            raise ValueError("MSM output channels must be divisible by 3.")
        branch = out_channels // 3

        def make(k: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ZeroPad2d((k // 2, k // 2, 0, 0)),
                nn.Conv2d(1, branch, kernel_size=(2, k)),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(branch),
            )

        self.conv_3 = make(3)
        self.conv_5 = make(5)
        self.conv_7 = make(7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.conv_3(x), self.conv_5(x), self.conv_7(x)], dim=1)


class TinyMLP(nn.Module):
    """Bottleneck MLP with Tanh output (official TinyMLP)."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n, n // 4),
            nn.ReLU(inplace=True),
            nn.Linear(n // 4, n),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AdaCorrModule(nn.Module):
    """ACM: learnable frequency-domain correction with a residual connection."""

    def __init__(self, length: int) -> None:
        super().__init__()
        self.re = TinyMLP(length)
        self.im = TinyMLP(length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computed in float32 so FFTs stay accurate under mixed-precision training.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            spec = torch.fft.fft(x, dim=-1)
            x_re, x_im = spec.real, spec.imag
            corrected = torch.complex(self.re(x_re) * x_re, self.im(x_im) * x_im)
            out = torch.fft.ifft(corrected, dim=-1).real
        return out + x


class FeaFusionModule(nn.Module):
    """FFM: multi-head self-attention across feature channels (official version)."""

    def __init__(self, num_heads: int, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be a multiple of num_heads.")
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.key = nn.Linear(input_size, hidden_size)
        self.query = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_size).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split(self.query(x))
        k = self._split(self.key(x))
        v = self._split(self.value(x))
        # Attention in float32: fp16 dot products over 256-dim heads can overflow.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = q.float(), k.float(), v.float()
            scores = (q / math.sqrt(self.head_size)) @ k.transpose(-2, -1)
            probs = self.dropout(F.softmax(scores, dim=-1))
            context = probs @ v
        b, h, t, d = context.shape
        return context.contiguous().view(b, h * t, d)     # same reshape as official code


class AMCNet(nn.Module):
    """ACM -> L2 norm -> MSM -> conv stem -> FFM -> GAP -> classifier.

    Adaptations for the EE6008 dataset:
    - input length 512 and 15 classes (paper: 128 and 11 on RML2016.10a);
      ACM and FFM sizes scale with input length exactly as in the official code;
    - ACM runs in float32 for mixed-precision safety;
    - epsilon added to the internal L2 normalisation to avoid division by zero.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 2,
        input_length: int = 128,
        hidden_channels: int = 36,
        dropout: float = 0.5,
        num_heads: int = 2,
        stem_channels: tuple[int, ...] | list[int] = (64, 128, 256),
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("AMC-Net expects 2 input channels (I and Q).")
        self.input_length = input_length
        self.acm = AdaCorrModule(input_length)
        self.msm = MultiScaleModule(hidden_channels)
        chans = [hidden_channels, *stem_channels]
        self.conv_stem = nn.Sequential(
            *(ConvBlock(chans[i], chans[i + 1]) for i in range(len(chans) - 1))
        )
        self.ffm = FeaFusionModule(num_heads, input_length, input_length, dropout)
        latent_dim = num_heads * chans[-1]
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] != self.input_length:
            raise ValueError(f"Expected [batch, 2, {self.input_length}], got {tuple(x.shape)}.")
        x = x.unsqueeze(1)                                              # [B, 1, 2, L]
        x = self.acm(x)
        x = x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        x = self.msm(x)                                                 # [B, 36, 1, L]
        x = self.conv_stem(x)                                           # [B, 256, 1, L]
        x = self.ffm(x.squeeze(2))                                      # [B, 512, L/2]
        x = self.gap(x).squeeze(2)                                      # [B, 512]
        return self.classifier(x)
