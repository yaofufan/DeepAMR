"""Build supported modulation recognition baselines."""

from torch import nn
from amc.config import ModelConfig
from .oshea_resnet import OSheaResNet
from .mcldnn import MCLDNN


def build_model(config: ModelConfig) -> nn.Module:
    models = {"oshea_resnet": OSheaResNet, "mcldnn": MCLDNN}
    if config.name.lower() not in models:
        raise ValueError(f"Unsupported model: {config.name}. Choose from {list(models)}.")
    return models[config.name.lower()](
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        input_length=config.input_length,
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
        **config.extra,
    )
