"""PyTorch reproduction of CCTL-Net for automatic modulation recognition.

Reference:
    W. Wang, X. Zou, Z. Pan, and H. Zhao, "A Complex-Valued Hybrid Deep
    Learning Models for Automatic Modulation Recognition," EURASIP JASP, 2025.
    DOI: https://doi.org/10.1186/s13634-025-01254-3

Implementation:
    Project PyTorch reproduction based on the paper architecture.

Input: I/Q tensors with shape [N, 2, L].
Output: Unnormalized class logits with shape [N, num_classes].
Note: Complex values use paired real and imaginary tensors.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.real = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.imag = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.real(real) - self.imag(imag),
            self.real(imag) + self.imag(real),
        )


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.real = nn.Linear(in_features, out_features)
        self.imag = nn.Linear(in_features, out_features)

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.real(real) - self.imag(imag),
            self.real(imag) + self.imag(real),
        )


class ComplexBatchNorm1d(nn.Module):
    """Whiten each complex channel using its 2x2 covariance matrix."""

    def __init__(self, channels: int, momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight_rr = nn.Parameter(torch.full((channels,), math.sqrt(0.5)))
        self.weight_ii = nn.Parameter(torch.full((channels,), math.sqrt(0.5)))
        self.weight_ri = nn.Parameter(torch.zeros(channels))
        self.bias_real = nn.Parameter(torch.zeros(channels))
        self.bias_imag = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(2, channels))
        running_covar = torch.zeros(3, channels)
        running_covar[:2].fill_(1.0)
        self.register_buffer("running_covar", running_covar)

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = real.dtype
        real32, imag32 = real.float(), imag.float()
        reduce_dims = (0, 2)
        if self.training:
            mean_real = real32.mean(reduce_dims)
            mean_imag = imag32.mean(reduce_dims)
            centered_real = real32 - mean_real[None, :, None]
            centered_imag = imag32 - mean_imag[None, :, None]
            cov_rr = centered_real.square().mean(reduce_dims) + self.eps
            cov_ii = centered_imag.square().mean(reduce_dims) + self.eps
            cov_ri = (centered_real * centered_imag).mean(reduce_dims)
            with torch.no_grad():
                mean = torch.stack((mean_real, mean_imag))
                covar = torch.stack((cov_rr, cov_ii, cov_ri))
                self.running_mean.lerp_(mean, self.momentum)
                self.running_covar.lerp_(covar, self.momentum)
        else:
            mean_real, mean_imag = self.running_mean
            cov_rr, cov_ii, cov_ri = self.running_covar
            centered_real = real32 - mean_real[None, :, None]
            centered_imag = imag32 - mean_imag[None, :, None]

        # Closed-form inverse square root of a positive 2x2 covariance matrix.
        determinant = (cov_rr * cov_ii - cov_ri.square()).clamp_min(self.eps)
        root_det = determinant.sqrt()
        scale = (cov_rr + cov_ii + 2.0 * root_det).clamp_min(self.eps).sqrt()
        denominator = (root_det * scale).clamp_min(self.eps)
        inv_rr = (cov_ii + root_det) / denominator
        inv_ii = (cov_rr + root_det) / denominator
        inv_ri = -cov_ri / denominator
        white_real = inv_rr[None, :, None] * centered_real + inv_ri[None, :, None] * centered_imag
        white_imag = inv_ri[None, :, None] * centered_real + inv_ii[None, :, None] * centered_imag

        out_real = (
            self.weight_rr[None, :, None] * white_real
            + self.weight_ri[None, :, None] * white_imag
            + self.bias_real[None, :, None]
        )
        out_imag = (
            self.weight_ri[None, :, None] * white_real
            + self.weight_ii[None, :, None] * white_imag
            + self.bias_imag[None, :, None]
        )
        return out_real.to(dtype), out_imag.to(dtype)


class ComplexSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = ComplexLinear(channels, reduced)
        self.fc2 = ComplexLinear(reduced, channels)
        self.prelu_real = nn.PReLU(reduced)
        self.prelu_imag = nn.PReLU(reduced)

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_real, gate_imag = self.fc1(real.mean(-1), imag.mean(-1))
        gate_real = self.prelu_real(gate_real)
        gate_imag = self.prelu_imag(gate_imag)
        gate_real, gate_imag = self.fc2(gate_real, gate_imag)
        gate_real = torch.sigmoid(gate_real).unsqueeze(-1)
        gate_imag = torch.sigmoid(gate_imag).unsqueeze(-1)
        return (
            gate_real * real - gate_imag * imag,
            gate_real * imag + gate_imag * real,
        )


class TalkingHeadAttention(nn.Module):
    """Shared multi-head attention with relative position and head mixing."""

    def __init__(self, dimension: int, heads: int, max_length: int, dropout: float) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("model_dim must be divisible by attention_heads.")
        self.heads = heads
        self.head_dim = dimension // heads
        self.max_length = max_length
        self.relative_position = nn.Parameter(torch.zeros(heads, 2 * max_length - 1))
        self.pre_softmax = nn.Parameter(torch.eye(heads))
        self.post_softmax = nn.Parameter(torch.eye(heads))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.relative_position, std=0.02)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch, length, dimension = query.shape
        if length > self.max_length:
            raise ValueError(f"Token length {length} exceeds configured maximum {self.max_length}.")
        query = query.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        positions = torch.arange(length, device=query.device)
        relative_index = positions[None, :] - positions[:, None] + self.max_length - 1
        scores = scores + self.relative_position[:, relative_index].unsqueeze(0)
        scores = torch.einsum("bhij,hk->bkij", scores, self.pre_softmax)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = torch.einsum("bhij,hk->bkij", weights, self.post_softmax)
        output = torch.matmul(self.dropout(weights), value)
        return output.transpose(1, 2).reshape(batch, length, dimension)


class ComplexTransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        max_length: int,
        ffn_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query = ComplexLinear(dimension, dimension)
        self.key = ComplexLinear(dimension, dimension)
        self.value = ComplexLinear(dimension, dimension)
        self.attention = TalkingHeadAttention(dimension, heads, max_length, dropout)
        self.output = ComplexLinear(dimension, dimension)
        self.norm1_real = nn.LayerNorm(dimension)
        self.norm1_imag = nn.LayerNorm(dimension)
        self.ffn_in = ComplexLinear(dimension, 2 * ffn_dimension)
        self.ffn_out = ComplexLinear(ffn_dimension, dimension)
        self.norm2_real = nn.LayerNorm(dimension)
        self.norm2_imag = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def _cmhsa(
        self,
        qr: torch.Tensor,
        qi: torch.Tensor,
        kr: torch.Tensor,
        ki: torch.Tensor,
        vr: torch.Tensor,
        vi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attend = self.attention
        real = (
            attend(qr, kr, vr)
            - attend(qr, ki, vi)
            - attend(qi, kr, vi)
            - attend(qi, ki, vr)
        )
        imag = (
            attend(qr, kr, vi)
            + attend(qr, ki, vr)
            + attend(qi, kr, vr)
            - attend(qi, ki, vi)
        )
        return real, imag

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qr, qi = self.query(real, imag)
        kr, ki = self.key(real, imag)
        vr, vi = self.value(real, imag)
        attn_real, attn_imag = self._cmhsa(qr, qi, kr, ki, vr, vi)
        attn_real, attn_imag = self.output(attn_real, attn_imag)
        real = self.norm1_real(real + self.dropout(attn_real))
        imag = self.norm1_imag(imag + self.dropout(attn_imag))

        ffn_real, ffn_imag = self.ffn_in(real, imag)
        value_real, gate_real = ffn_real.chunk(2, dim=-1)
        value_imag, gate_imag = ffn_imag.chunk(2, dim=-1)
        ffn_real = value_real * F.relu(gate_real)
        ffn_imag = value_imag * F.relu(gate_imag)
        ffn_real, ffn_imag = self.ffn_out(ffn_real, ffn_imag)
        return (
            self.norm2_real(real + self.dropout(ffn_real)),
            self.norm2_imag(imag + self.dropout(ffn_imag)),
        )


class CCTLNet(nn.Module):
    """Complex CNN, complex Transformer, and complex LSTM hybrid network."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 2,
        input_length: int = 128,
        hidden_channels: int = 64,
        dropout: float = 0.1,
        token_length: int = 128,
        kernel_size: int = 3,
        attention_heads: int = 8,
        transformer_layers: int = 2,
        ffn_dimension: int = 128,
        lstm_hidden: int = 128,
        lstm_paths: int = 4,
        se_reduction: int = 4,
        fc_channels: int = 128,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("CCTL-Net requires two I/Q input channels.")
        if token_length > input_length:
            raise ValueError("token_length cannot exceed input_length.")
        if lstm_paths != 4:
            raise ValueError("The paper configuration uses four complex-LSTM paths.")
        self.input_length = input_length
        self.token_length = token_length
        self.embedding = ComplexConv1d(1, hidden_channels, kernel_size)
        self.embedding_norm = ComplexBatchNorm1d(hidden_channels)
        self.prelu_real = nn.PReLU(hidden_channels)
        self.prelu_imag = nn.PReLU(hidden_channels)
        self.se = ComplexSE(hidden_channels, se_reduction)
        self.transformer = nn.ModuleList(
            ComplexTransformerBlock(
                hidden_channels,
                attention_heads,
                token_length,
                ffn_dimension,
                dropout,
            )
            for _ in range(transformer_layers)
        )
        # Equation (18) uses two learned real LSTMs in four shared-weight paths.
        self.lstm_real = nn.LSTM(hidden_channels, lstm_hidden, batch_first=True)
        self.lstm_imag = nn.LSTM(hidden_channels, lstm_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, fc_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_channels, fc_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_channels, num_classes),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (2, self.input_length):
            raise ValueError(f"Expected [N, 2, {self.input_length}], got {tuple(x.shape)}.")
        if self.input_length != self.token_length:
            x = F.adaptive_avg_pool1d(x, self.token_length)
        real, imag = self.embedding(x[:, 0:1], x[:, 1:2])
        real, imag = self.embedding_norm(real, imag)
        real = self.prelu_real(real)
        imag = self.prelu_imag(imag)
        real, imag = self.se(real, imag)
        real, imag = real.transpose(1, 2), imag.transpose(1, 2)
        for block in self.transformer:
            real, imag = block(real, imag)

        rr, _ = self.lstm_real(real)
        ir, _ = self.lstm_real(imag)
        ri, _ = self.lstm_imag(real)
        ii, _ = self.lstm_imag(imag)
        output_real = rr[:, -1] - ii[:, -1]
        output_imag = ir[:, -1] + ri[:, -1]
        return self.classifier(torch.cat((output_real, output_imag), dim=-1))
