"""Signal components used by the best SAFTFormer configuration.

DB-GLU: https://doi.org/10.1109/LCOMM.2022.3213523
Temporal gating: https://arxiv.org/abs/2311.17132
Mixed temporal kernels: https://arxiv.org/abs/2001.06769
The spectral path reads frame-local complex-IQ power spectra.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


def temporal_kernels(values: Sequence[int], name: str = "temporal_kernel_sizes") -> tuple[int, ...]:
    if (not isinstance(values, Sequence) or isinstance(values, (str, bytes))
            or not values or any(not isinstance(k, int) or isinstance(k, bool)
                                 or k < 1 or k % 2 == 0 for k in values)):
        raise ValueError(f"{name} must be a nonempty sequence of positive odd integers.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique scales.")
    return tuple(values)


class MixedTemporalDWConv(nn.Module):
    """Split temporal channels across the configured odd kernels and concatenate."""

    def __init__(self, channels: int, kernel_sizes: Sequence[int]) -> None:
        super().__init__()
        self.kernel_sizes = temporal_kernels(kernel_sizes)
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < len(self.kernel_sizes):
            raise ValueError("Temporal branch width must provide at least one channel per scale.")
        self.channels = channels
        width, remainder = divmod(channels, len(self.kernel_sizes))
        self.split_sizes = tuple(width + int(i < remainder) for i in range(len(self.kernel_sizes)))
        self.convs = nn.ModuleList([
            nn.Conv1d(size, size, kernel, padding=kernel // 2, groups=size, bias=False)
            for size, kernel in zip(self.split_sizes, self.kernel_sizes)
        ])
        for conv in self.convs:
            nn.init.zeros_(conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"Expected temporal features [B,{self.channels},time].")
        return torch.cat([conv(group) for conv, group in
                          zip(self.convs, x.split(self.split_sizes, dim=1))], dim=1)


class DualBranchGLU(nn.Module):
    """Symmetric a*activation(b) + b*activation(a), then output projection."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.activation = nn.SiLU()
        self.a = nn.Linear(dim, hidden_dim)
        self.b = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.hidden_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.a(x), self.b(x)
        hidden = a * self.activation(b) + b * self.activation(a)
        return self.out_drop(self.out(self.hidden_drop(hidden)))


class TimeFrequencyDBGLU(DualBranchGLU):
    """Condition both SiLU gates with temporal and spectral features."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float,
                 temporal_kernel_sizes: Sequence[int]) -> None:
        super().__init__(dim, hidden_dim, dropout)
        self.temporal = MixedTemporalDWConv(hidden_dim, temporal_kernel_sizes)
        self.frequency_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor,
                frequency_context: torch.Tensor | None = None) -> torch.Tensor:
        a, b = self.a(x), self.b(x)
        gate_a, gate_b = a, b
        # Tokens are ordered in time; the same kernel acts on both branches.
        gate_a = gate_a + self.temporal(a.transpose(1, 2)).transpose(1, 2)
        gate_b = gate_b + self.temporal(b.transpose(1, 2)).transpose(1, 2)
        if frequency_context is None or frequency_context.shape != a.shape:
            raise ValueError("Frequency conditioning requires aligned [B,tokens,branch_width] features.")
        condition = self.frequency_scale.to(a.dtype) * frequency_context.to(a.dtype)
        gate_a, gate_b = gate_a + condition, gate_b + condition
        hidden = a * self.activation(gate_b) + b * self.activation(gate_a)
        return self.out_drop(self.out(self.hidden_drop(hidden)))


class FrameSpectralConditioner(nn.Module):
    """Hann-windowed complex FFT, normalized log power, PW-DW-PW, projection."""

    def __init__(self, frame_length: int, frame_stride: int, hidden_dim: int,
                 channels: int, kernel_size: int, eps: float) -> None:
        super().__init__()
        self.frame_length, self.frame_stride, self.eps = frame_length, frame_stride, eps
        self.register_buffer("window", torch.hann_window(frame_length), persistent=False)
        pointwise_in = nn.Conv1d(1, channels, 1, bias=False)
        frequency_dwconv = nn.Conv1d(channels, channels, kernel_size,
                                    padding=kernel_size // 2, groups=channels, bias=False)
        self.frequency_conv = nn.Sequential(
            pointwise_in, nn.SiLU(), frequency_dwconv, nn.SiLU(),
            nn.Conv1d(channels, 1, 1, bias=False))
        self.projection = nn.Linear(frame_length, hidden_dim, bias=False)

    def log_power_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        spectrum_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            raw = x.to(spectrum_dtype)
            i = raw[:, 0].unfold(-1, self.frame_length, self.frame_stride)
            q = raw[:, 1].unfold(-1, self.frame_length, self.frame_stride)
            signal = torch.complex(i, q) * self.window.to(spectrum_dtype)
            # Complex baseband has no general conjugate symmetry: retain all bins.
            spectrum = torch.fft.fftshift(torch.fft.fft(signal, norm="ortho"), dim=-1)
            power = spectrum.real.square() + spectrum.imag.square()
            relative_power = power / (power.mean(dim=-1, keepdim=True) + self.eps)
            return (relative_power + self.eps).log()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.log_power_spectrum(x)
        batch, tokens, bins = features.shape
        features = self.frequency_conv(features.reshape(batch * tokens, 1, bins))
        return self.projection(features.reshape(batch, tokens, bins))


class SymbolHierarchicalEmbedding(nn.Module):
    """Encode eight-sample IQ windows and project six-symbol frames."""

    def __init__(self, frame_length: int, frame_stride: int, dim: int,
                 symbol_length: int = 8, symbol_channels: int = 16) -> None:
        super().__init__()
        self.frame_length, self.frame_stride = frame_length, frame_stride
        self.symbol_length, self.symbol_channels = symbol_length, symbol_channels
        self.symbol_encoder = nn.Conv1d(2, symbol_channels, symbol_length,
                                       stride=symbol_length, bias=False)
        self.activation = nn.SiLU()
        self.frame_projection = nn.Conv1d(symbol_channels, dim, frame_length // symbol_length,
                                         stride=frame_stride // symbol_length, bias=False)
        nn.init.xavier_uniform_(self.symbol_encoder.weight)
        nn.init.xavier_uniform_(self.frame_projection.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[-1] < self.frame_length:
            raise ValueError(f"SHE expects [B,2,L] with L >= {self.frame_length}.")
        symbols = self.activation(self.symbol_encoder(x))
        return self.frame_projection(symbols).transpose(1, 2)
