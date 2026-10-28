"""PyTorch reproduction of TLDNN for automatic modulation recognition.

Reference:
    Y. Qu et al., "Enhancing Automatic Modulation Recognition through
    Robust Global Feature Extraction," arXiv, 2024.
    Paper: https://arxiv.org/abs/2401.01056

Implementation:
    Group-member PyTorch reproduction of TransformerLSTM_RML2018.

Input: I/Q tensors with shape [N, 2, L].
Output: Unnormalized class logits with shape [N, num_classes].
Note: Convolutional reduction, talking-head attention and LSTM readout.
"""

import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x).squeeze(-1)).unsqueeze(-1)
        return x * scale


class TalkingHeadsAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("d_model must be divisible by nhead")
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.mix_logits = nn.Conv2d(heads, heads, 1, bias=False)
        self.mix_weights = nn.Conv2d(heads, heads, 1, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, dim = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, steps, self.heads, self.dim_head).transpose(1, 2)
        q, k, v = map(split_heads, (q, k, v))
        scores = self.mix_logits(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        weights = self.attention_dropout(scores.softmax(dim=-1))
        weights = self.mix_weights(weights)
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous().reshape(batch, steps, dim)
        return self.to_out(attended)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = TalkingHeadsAttention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.glu_left = nn.Linear(dim, mlp_dim)
        self.glu_right = nn.Linear(dim, mlp_dim)
        self.glu_out = nn.Sequential(nn.LayerNorm(mlp_dim), nn.Dropout(dropout), nn.Linear(mlp_dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attention(x))
        gated = self.activation(self.glu_left(x)) * self.glu_right(x)
        return self.norm2(x + self.glu_out(gated))


class TLDNN(nn.Module):
    """TLDNN with DeepAMR-compatible constructor and output shape."""
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 2,
        input_length: int = 512,
        dropout: float = 0.2,
        d_model: int = 64,
        nhead: int = 8,
        d_hid: int = 128,
        lstm_hidden_size: int = 64,
        nlayers: int = 2,
        **_: object,
    ) -> None:
        super().__init__()
        if input_channels != 2 or input_length < 16:
            raise ValueError("TLDNN requires two I/Q channels and input_length >= 16.")
        if d_model % 4:
            raise ValueError("d_model must be divisible by 4 for squeeze-excitation.")
        self.input_length = input_length
        self.convlayer = nn.Sequential(
            nn.Conv1d(2, d_model, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, d_model))
        self.se = SqueezeExcitation(d_model, reduction=4)
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model, nhead, d_hid, dropout) for _ in range(nlayers)
        ])
        self.lstm = nn.LSTM(d_model, lstm_hidden_size, num_layers=4, dropout=dropout, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size), nn.ReLU(),
            nn.BatchNorm1d(lstm_hidden_size), nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2), nn.ReLU(),
            nn.BatchNorm1d(lstm_hidden_size // 2), nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (2, self.input_length):
            raise ValueError(f"Expected [N, 2, {self.input_length}], got {tuple(x.shape)}.")
        x = self.convlayer(x.float()).transpose(1, 2)
        steps = x.shape[1]
        if steps > self.pos_embedding.shape[1]:
            position = torch.nn.functional.interpolate(
                self.pos_embedding.transpose(1, 2), size=steps, mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            position = self.pos_embedding[:, :steps]
        x = self.se((x + position).transpose(1, 2)).transpose(1, 2)
        x = self.transformer(x)
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])
