"""PyTorch reproduction of MCformer for automatic modulation recognition.

Reference:
    S. Hamidi-Rad and S. Jain, "MCformer: A Transformer Based Deep Neural
    Network for Automatic Modulation Classification," GLOBECOM, 2021.
    DOI: https://doi.org/10.1109/GLOBECOM46510.2021.9685815

Implementation:
    Group-member PyTorch reproduction adapted to the AMC interface.

Input: I/Q tensors with shape [N, 2, L].
Output: Unnormalized class logits with shape [N, num_classes].
Note: Convolutional embedding, Transformer encoder and token readout.
"""

from torch import nn


class MCformer(nn.Module):
    def __init__(self, num_classes=15, input_channels=2, input_length=512,
                 hidden_channels=32, dropout=0.1, kernel_size=65, num_layers=4,
                 num_heads=4, pool_vectors=4, dense_units=128, **kwargs):
        super().__init__()
        self.pool_vectors = pool_vectors
        self.embed = nn.Conv1d(input_channels, hidden_channels, kernel_size,
                               padding=kernel_size // 2)
        self.act = nn.SELU()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels, nhead=num_heads,
            dim_feedforward=4 * hidden_channels, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.Linear(pool_vectors * hidden_channels, dense_units),
            nn.SELU(),
            nn.Linear(dense_units, num_classes))

    def forward(self, x):
        x = self.act(self.embed(x))
        x = self.encoder(x.transpose(1, 2))
        x = x[:, :self.pool_vectors, :].flatten(1)
        return self.classifier(x)
