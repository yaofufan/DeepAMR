"""PyTorch reproduction of DHN for automatic modulation recognition.

Reference:
    J. Nie et al., "Deep Hierarchical Network for Automatic
    Modulation Classification," IEEE Access, 2019.
    DOI: https://doi.org/10.1109/ACCESS.2019.2928463

Implementation:
    Group-member PyTorch reproduction based on the paper architecture.

Input: I/Q tensors with shape [N, 2, L].
Output: Unnormalized class logits with shape [N, num_classes].
Note: SNR-weighted training loss is implemented in amc.training.engine.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SameConv2d(nn.Module):
    """TensorFlow-style SAME padding, including even kernels."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.conv = nn.Conv2d(in_channels, out_channels, self.kernel_size, self.stride, padding=0, bias=bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        out_h = (h + self.stride[0] - 1) // self.stride[0]
        out_w = (w + self.stride[1] - 1) // self.stride[1]
        pad_h = max(0, (out_h - 1) * self.stride[0] + self.kernel_size[0] - h)
        pad_w = max(0, (out_w - 1) * self.stride[1] + self.kernel_size[1] - w)
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        return self.conv(x)


class DHNBottleneck(nn.Module):
    def __init__(self, in_channels, filters, stride=1):
        super().__init__()
        f1, f2, f3 = filters
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.conv2 = SameConv2d(f1, f2, kernel_size=(2, 4), stride=(1, stride), bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(f3)
        self.activation = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or in_channels != f3:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, f3, kernel_size=1, stride=(1, stride), bias=False),
                nn.BatchNorm2d(f3),
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.activation(x + identity)


class FeatureAttention(nn.Module):
    """Learn one data-dependent gate per concatenated feature channel."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.fc(self.pool(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class DHN(nn.Module):
    """DHN backbone returning class logits for [N,2,L] I/Q inputs."""
    # Number of bottleneck blocks follows Table I: 1+2+1+3+1+5+1+2 = 16.
    STAGES = (
        ((2, 4, 8), 1),
        ((2, 4, 8), 2),
        ((4, 8, 16), 1),
        ((4, 8, 16), 3),
        ((8, 16, 32), 1),
        ((8, 16, 32), 5),
        ((16, 32, 64), 1),
        ((16, 32, 64), 2),
    )

    def __init__(self, num_classes, input_channels=2, input_length=512,
                 hidden_channels=64, dropout=0.0, attention_reduction=8, **kwargs):
        super().__init__()
        if input_channels != 2 or input_length < 16:
            raise ValueError("DHN requires I/Q input with input_length >= 16")
        self.input_length = input_length
        # Pad the I/Q axis from 2 to 4 as in the paper's input diagram.
        self.stem = nn.Sequential(
            SameConv2d(1, 64, kernel_size=(1, 4)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        in_channels = 64
        stages = []
        for stage_idx, (filters, block_count) in enumerate(self.STAGES):
            blocks = []
            for block_idx in range(block_count):
                # Progressive temporal reduction creates the global high-level map.
                stride = 2 if stage_idx in {1, 3, 5, 7} and block_idx == 0 else 1
                blocks.append(DHNBottleneck(in_channels, filters, stride=stride))
                in_channels = filters[2]
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        # The paper fuses the fourth convolutional layer with the final map.
        # Capture the first bottleneck output as the shallow local representation.
        self.shallow_channels = self.STAGES[0][0][2]
        self.deep_channels = in_channels
        fused_channels = self.shallow_channels + self.deep_channels
        self.feature_attention = FeatureAttention(fused_channels, attention_reduction)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.classifier = nn.Linear(fused_channels, num_classes)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (2, self.input_length):
            raise ValueError(f"Expected [N, 2, {self.input_length}], got {tuple(x.shape)}")
        # [N,2,L] -> [N,1,4,L] with zero padding on the signal-component axis.
        x = F.pad(x.float().unsqueeze(1), (0, 0, 1, 1))
        x = self.stem(x)
        shallow = None
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if stage_idx == 0:
                shallow = x
        shallow = F.adaptive_avg_pool2d(shallow, x.shape[-2:])
        fused = torch.cat((shallow, x), dim=1)
        fused = self.feature_attention(fused)
        fused = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        return self.classifier(self.dropout(fused))
