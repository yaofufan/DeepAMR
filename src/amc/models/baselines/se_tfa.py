"""PyTorch reproduction of SE-TFA for automatic modulation recognition.

Reference:
    S. Lin, Y. Zeng, and Y. Gong, "Modulation Recognition Using Signal
    Enhancement and Multistage Attention Mechanism," IEEE TWC, 2022.
    DOI: https://doi.org/10.1109/TWC.2022.3181026

Implementation:
    Project PyTorch reproduction based on the paper architecture.

Input: I/Q tensors with shape [N, 2, L].
Output: Unnormalized class logits with shape [N, num_classes].
Note: Uses the separate three-stage SE-TFA training protocol.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _same_padding(kernel_size: tuple[int, int], dilation: tuple[int, int]) -> tuple[int, int]:
    return tuple(d * (k - 1) // 2 for k, d in zip(kernel_size, dilation))


class TimeFrequencyAttention(nn.Module):
    """Channel, time, and frequency attention block from Figure 6."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )
        self.time_attention = self._axis_attention()
        self.frequency_attention = self._axis_attention()
        self.fuse = nn.Conv2d(2 * channels, channels, 1)

    @staticmethod
    def _axis_attention() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = x.mean(dim=(2, 3))
        maximum = x.amax(dim=(2, 3))
        channel_gate = torch.sigmoid(
            self.channel_mlp(average) + self.channel_mlp(maximum)
        ).unsqueeze(-1).unsqueeze(-1)
        channel_refined = x * channel_gate

        channel_average = channel_refined.mean(dim=1, keepdim=True)
        channel_maximum = channel_refined.amax(dim=1, keepdim=True)
        descriptor = torch.cat((channel_average, channel_maximum), dim=1)
        time_descriptor = descriptor.mean(dim=3, keepdim=True)
        frequency_descriptor = descriptor.mean(dim=2, keepdim=True)
        time_refined = channel_refined * self.time_attention(time_descriptor)
        frequency_refined = channel_refined * self.frequency_attention(frequency_descriptor)
        return self.fuse(torch.cat((time_refined, frequency_refined), dim=1))


class DilatedTFAStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        dilation: tuple[int, int],
        reduction: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=_same_padding(kernel_size, dilation),
            dilation=dilation,
        )
        self.attention = TimeFrequencyAttention(out_channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.attention(self.conv(x)), inplace=True)


class SignalEnhancement(nn.Module):
    """Eleven-layer ratio-mask estimator from Table I."""

    def __init__(self, input_channels: int = 6, reduction: int = 4) -> None:
        super().__init__()
        specifications = [
            ((1, 7), (1, 1)),
            ((7, 1), (1, 1)),
            ((5, 5), (1, 1)),
            ((5, 5), (2, 1)),
            ((5, 5), (4, 1)),
            ((5, 5), (8, 1)),
            ((5, 5), (1, 1)),
            ((5, 5), (2, 2)),
            ((5, 5), (4, 4)),
            ((5, 5), (8, 8)),
        ]
        stages = []
        channels = input_channels
        for kernel, dilation in specifications:
            stages.append(DilatedTFAStage(channels, 48, kernel, dilation, reduction))
            channels = 48
        self.stages = nn.Sequential(*stages)
        self.mask = nn.Sequential(nn.Conv2d(48, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask(self.stages(x))


class ClassificationStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.attention = TimeFrequencyAttention(out_channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.attention(self.conv(x)), inplace=True)


class SETFA(nn.Module):
    """Dual-spectrum fusion, enhancement, and multistage TFA classifier."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 2,
        input_length: int = 128,
        hidden_channels: int = 48,
        dropout: float = 0.0,
        spectrogram_size: int = 100,
        frame_length: int = 40,
        frame_shift: int = 2,
        fft_size: int = 128,
        ma_window: int = 9,
        gaussian_sigma: float = 1.0,
        gaussian_radius: int = 7,
        attention_reduction: int = 4,
        fc_channels: int = 128,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("SE-TFA requires two I/Q input channels.")
        if hidden_channels != 48:
            raise ValueError("Table I specifies 48 channels in the enhancement module.")
        if spectrogram_size < 16:
            raise ValueError("spectrogram_size must be at least 16.")
        self.input_length = input_length
        self.spectrogram_size = spectrogram_size
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.fft_size = fft_size
        self.ma_window = ma_window
        self.gaussian_radius = gaussian_radius
        self.register_buffer("stft_window", torch.hamming_window(frame_length), persistent=False)
        positions = torch.arange(-gaussian_radius, gaussian_radius + 1, dtype=torch.float32)
        gaussian = torch.exp(-positions.square() / (2.0 * gaussian_sigma**2))
        self.register_buffer("gaussian_kernel", gaussian / gaussian.sum(), persistent=False)

        self.enhancement = SignalEnhancement(6, attention_reduction)
        self.classification_stages = nn.ModuleList(
            [
                ClassificationStage(6, 64, attention_reduction),
                ClassificationStage(64, 32, attention_reduction),
                ClassificationStage(32, 12, attention_reduction),
                ClassificationStage(12, 8, attention_reduction),
            ]
        )
        pooled_size = spectrogram_size
        for _ in range(3):
            pooled_size //= 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * pooled_size * pooled_size, fc_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_channels, num_classes),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _filter_signal(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        moving = torch.ones(channels, 1, self.ma_window, device=x.device, dtype=x.dtype)
        moving = moving / self.ma_window
        x = F.conv1d(x, moving, padding=self.ma_window // 2, groups=channels)
        gaussian = self.gaussian_kernel.to(device=x.device, dtype=x.dtype)
        gaussian = gaussian.view(1, 1, -1).expand(channels, 1, -1)
        return F.conv1d(x, gaussian, padding=self.gaussian_radius, groups=channels)

    @staticmethod
    def _jet(values: torch.Tensor) -> torch.Tensor:
        def channel(center: float) -> torch.Tensor:
            return (1.5 - (4.0 * values - center).abs()).clamp(0.0, 1.0)

        return torch.stack((channel(3.0), channel(2.0), channel(1.0)), dim=1)

    def _spectrogram_rgb(self, x: torch.Tensor) -> torch.Tensor:
        signal = torch.complex(x[:, 0].float(), x[:, 1].float())
        spectrum = torch.stft(
            signal,
            n_fft=self.fft_size,
            hop_length=self.frame_shift,
            win_length=self.frame_length,
            window=self.stft_window.float(),
            center=True,
            return_complex=True,
        ).abs()
        spectrum = torch.log1p(spectrum)
        minimum = spectrum.amin(dim=(1, 2), keepdim=True)
        maximum = spectrum.amax(dim=(1, 2), keepdim=True)
        normalized = (spectrum - minimum) / (maximum - minimum).clamp_min(1e-6)
        return self._jet(normalized)

    def _dual_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        # cuFFT/STFT is kept in FP32 because complex half support is incomplete.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            original = self._spectrogram_rgb(x.float())
            filtered = self._spectrogram_rgb(self._filter_signal(x.float()))
            fused = torch.cat((original, filtered), dim=1)
            return F.interpolate(
                fused,
                size=(self.spectrogram_size, self.spectrogram_size),
                mode="bilinear",
                align_corners=False,
            )

    def classify_spectrum(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the signal-classification module from Table II."""
        for index, stage in enumerate(self.classification_stages):
            features = stage(features)
            if index < 3:
                features = F.max_pool2d(features, 2)
        return self.classifier(features)

    def forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-train SC on unenhanced dual-spectrum features as in the paper."""
        self._validate_input(x)
        return self.classify_spectrum(self._dual_spectrum(x))

    def set_training_stage(self, stage: str) -> None:
        """Select trainable parameters for the three-stage paper protocol."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        if stage == "classifier_pretrain":
            modules = [self.classification_stages, self.classifier]
        elif stage == "enhancement":
            modules = [self.enhancement]
        elif stage == "joint_finetune":
            modules = [self]
        else:
            raise ValueError(f"Unknown SE-TFA training stage: {stage}")
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or tuple(x.shape[1:]) != (2, self.input_length):
            raise ValueError(f"Expected [N, 2, {self.input_length}], got {tuple(x.shape)}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        features = self.enhancement(self._dual_spectrum(x))
        return self.classify_spectrum(features)
