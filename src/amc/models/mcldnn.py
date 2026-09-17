"""PyTorch implementation of Xu et al.'s MCLDNN (IEEE WCL, 2020).

Architecture reference: https://github.com/wzjialang/MCLDNN/blob/master/MCLDNN.py
Accepts [N, 2, L] I/Q tensors and returns unnormalized class logits.
"""

import torch
from torch import nn
from torch.nn import functional as F


class MCLDNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 2,
        input_length: int = 128,
        hidden_channels: int = 50,
        dropout: float = 0.5,
        lstm_hidden: int = 128,
        fc_channels: int = 128,
    ) -> None:
        super().__init__()
        if input_channels != 2 or input_length < 5:
            raise ValueError("MCLDNN requires two I/Q channels and input_length >= 5.")
        self.input_length = input_length
        self.iq_conv = nn.Conv2d(1, hidden_channels, (2, 8))
        self.i_conv = nn.Conv1d(1, hidden_channels, 8)
        self.q_conv = nn.Conv1d(1, hidden_channels, 8)
        self.separate_conv = nn.Conv2d(hidden_channels, hidden_channels, (1, 8))
        self.fusion_conv = nn.Conv2d(2 * hidden_channels, 100, (2, 5))
        self.lstm = nn.LSTM(100, lstm_hidden, num_layers=2, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, fc_channels),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_channels, fc_channels),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_channels, num_classes),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for name, value in self.lstm.named_parameters():
            if name.startswith("weight_ih"):
                nn.init.xavier_uniform_(value)
            elif name.startswith("weight_hh"):
                nn.init.orthogonal_(value)
            else:
                nn.init.zeros_(value)
                if name.startswith("bias_ih"):
                    # Keras unit_forget_bias=True: total forget bias is one.
                    hidden = self.lstm.hidden_size
                    with torch.no_grad():
                        value[hidden:2 * hidden].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (2, self.input_length):
            raise ValueError(f"Expected [N, 2, {self.input_length}], got {tuple(x.shape)}.")
        # TensorFlow SAME puts the extra pad on the right/bottom for even kernels.
        joint = F.relu(self.iq_conv(F.pad(x.unsqueeze(1), (3, 4, 0, 1))))
        i = F.relu(self.i_conv(F.pad(x[:, 0:1, :], (7, 0))))
        q = F.relu(self.q_conv(F.pad(x[:, 1:2, :], (7, 0))))
        separate = torch.stack((i, q), dim=2)
        separate = F.relu(self.separate_conv(F.pad(separate, (3, 4, 0, 0))))
        fused = F.relu(self.fusion_conv(torch.cat((joint, separate), dim=1)))
        sequence = fused.squeeze(2).transpose(1, 2).contiguous()
        sequence, _ = self.lstm(sequence)
        return self.classifier(sequence[:, -1, :])
