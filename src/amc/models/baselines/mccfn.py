"""PyTorch reproduction of MCCFN for automatic modulation recognition.

Reference:
    A. Li et al., "A Multi-scale Complex-valued Convolutional Fusion Network
    for Automatic Modulation Recognition," Mobile Networks and Applications, 2026.
    DOI: https://doi.org/10.1007/s11036-026-02513-9
    Code: https://github.com/Miraitwo/MCCFN

Implementation:
    Group-member port of the official MCCFN implementation.

Input: I/Q tensors with shape [N, 2, L].
Output: Unnormalized class logits with shape [N, num_classes].
Note: Complex convolutions use paired real and imaginary tensors.
"""

from __future__ import annotations

import torch
from torch import nn


class ComplexConv1d1(nn.Module):
    """Complex-valued convolution used by the convolution stem."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        complex_axis=1,
    ):
        super().__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups // 2 if groups == in_channels else 1
        self.dilation = dilation
        self.bias = bias
        self.complex_axis = complex_axis

        self.real_conv = nn.Conv1d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation,
            groups=self.groups, bias=self.bias
        )
        self.imag_conv = nn.Conv1d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation,
            groups=self.groups, bias=self.bias
        )
        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)

    def forward(self, inputs):
        inputs = inputs.squeeze(2)
        real, imag = torch.chunk(inputs, 2, self.complex_axis)

        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)
        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        return torch.cat([real, imag], self.complex_axis).unsqueeze(2)


class Conv_Block(nn.Module):
    """Complex convolution + activation + batch normalization."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            ComplexConv1d1(self._check_even(in_channel), self._check_even(out_channel), kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel),
        )

    @staticmethod
    def _check_even(value):
        if value % 2 != 0:
            raise ValueError(f"Complex channel count must be even, got {value}.")
        return value

    def forward(self, x):
        return self.conv_block(x)


class ComplexConv1d(nn.Module):
    """Complex-valued convolution used by the multi-scale module."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        complex_axis=1,
    ):
        super().__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups // 2 if groups == in_channels else 1
        self.dilation = dilation
        self.bias = bias
        self.complex_axis = complex_axis

        self.real_conv = nn.Conv1d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation,
            groups=self.groups, bias=self.bias
        )
        self.imag_conv = nn.Conv1d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation,
            groups=self.groups, bias=self.bias
        )
        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)

    def forward(self, inputs):
        inputs = inputs.squeeze(1)
        real, imag = torch.chunk(inputs, 2, self.complex_axis)

        real2real = self.real_conv(real)
        imag2imag = self.imag_conv(imag)
        real2imag = self.imag_conv(real)
        imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        return torch.cat([real, imag], self.complex_axis).unsqueeze(2)


class CReLU(nn.Module):
    """Complex-valued ReLU retained from the official implementation."""

    def __init__(self, complex_axis=1, inplace=False):
        super().__init__()
        self.r_relu = nn.ReLU(inplace=inplace)
        self.i_relu = nn.ReLU(inplace=inplace)
        self.complex_axis = complex_axis

    def forward(self, inputs):
        inputs = inputs.squeeze(2)
        real, imag = torch.chunk(inputs, 2, self.complex_axis)
        output = torch.cat([self.r_relu(real), self.i_relu(imag)], self.complex_axis)
        return output.unsqueeze(2)


class MultiScaleModule(nn.Module):
    """Three complex convolution branches with kernels 3, 5 and 7."""

    def __init__(self, out_channel):
        super().__init__()
        self.out_c = out_channel

        self.conv_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            ComplexConv1d(2, self.out_c // 3, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3),
        )
        self.conv_5 = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            ComplexConv1d(2, self.out_c // 3, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3),
        )
        self.conv_7 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            ComplexConv1d(2, self.out_c // 3, kernel_size=7),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3),
        )

    def forward(self, x):
        return torch.cat([self.conv_3(x), self.conv_5(x), self.conv_7(x)], dim=1)


class FeaFusionModule(nn.Module):
    """Feature fusion module using multi-head self-attention."""

    def __init__(self, num_attention_heads, input_size, hidden_size):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden size {hidden_size} is not divisible by "
                f"attention heads {num_attention_heads}"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=0.5,
        )
        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Official MCCFN layout: [batch, 256, signal_length].
        batch_size, seq_len, _ = x.size()

        key = self.key_layer(x).permute(1, 0, 2)
        query = self.query_layer(x).permute(1, 0, 2)
        value = self.value_layer(x).permute(1, 0, 2)

        attn_output, _ = self.attention(query, key, value)
        attn_output = self.dropout(attn_output)
        return attn_output.permute(1, 0, 2)


class MCCFN(nn.Module):
    """MCCFN adapted to the DeepAMR [N, 2, L] model interface."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 2,
        input_length: int = 128,
        hidden_channels: int = 36,
        dropout: float = 0.0,
        sig_len: int | None = None,
        extend_channel: int | None = None,
        latent_dim: int = 512,
        num_heads: int = 2,
        conv_chan_list: list[int] | None = None,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("MCCFN requires exactly two input channels: I and Q.")
        if input_length < 7:
            raise ValueError("MCCFN requires input_length >= 7.")
        if sig_len is not None and sig_len != input_length:
            raise ValueError(
                f"sig_len={sig_len} must match input_length={input_length}."
            )

        self.sig_len = input_length
        self.extend_channel = hidden_channels if extend_channel is None else extend_channel
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.conv_chan_list = (
            [36, 64, 128, 256] if conv_chan_list is None else list(conv_chan_list)
        )

        if self.conv_chan_list[0] != self.extend_channel:
            raise ValueError(
                "The first conv_chan_list value must equal extend_channel "
                f"({self.extend_channel})."
            )
        if any(channel % 2 for channel in self.conv_chan_list):
            raise ValueError("All complex convolution channel counts must be even.")

        self.MSM = MultiScaleModule(self.extend_channel)
        self.FFM1 = FeaFusionModule(
            self.num_heads,
            self.sig_len,
            self.sig_len,
        )

        self.Conv_stem = nn.Sequential()
        for index in range(len(self.conv_chan_list) - 1):
            self.Conv_stem.add_module(
                f"conv_stem_{index}",
                Conv_Block(
                    self.conv_chan_list[index],
                    self.conv_chan_list[index + 1],
                ),
            )

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, self.latent_dim),
            nn.Dropout(0.0),  # official MCCFN value
            nn.PReLU(),
            nn.Linear(self.latent_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (2, self.sig_len):
            raise ValueError(
                f"Expected [N, 2, {self.sig_len}], got {tuple(x.shape)}."
            )

        x = x.unsqueeze(1)
        x = self.MSM(x)
        x = self.Conv_stem(x)
        x = self.FFM1(x.squeeze(2))
        x = self.GAP(x)
        return self.classifier(x.squeeze(2))
