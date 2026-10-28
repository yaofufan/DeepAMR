"""Build the research model and supported modulation recognition baselines."""

from torch import nn
from amc.config import ModelConfig
from .baselines import AMCNet, CCTLNet, DHN, MCCFN, MCformer, MCLDNN, OSheaResNet, SETFA, TLDNN
from .ours_saftformer import ViTSPlus1D


def build_model(config: ModelConfig) -> nn.Module:
    models = {
        "oshea_resnet": OSheaResNet,
        "mcldnn": MCLDNN,
        "cctl_net": CCTLNet,
        "se_tfa": SETFA,
        "amcnet": AMCNet,
        "mcformer": MCformer,
        "mccfn": MCCFN,
        "tldnn": TLDNN,
        "dhn": DHN,
        "vit_splus_1d": ViTSPlus1D,
    }
    # An optional, local-only extension can register private experiments.
    try:
        from .local_models import LOCAL_MODELS
    except ModuleNotFoundError as exc:
        if exc.name != f"{__package__}.local_models":
            raise
    else:
        models.update(LOCAL_MODELS)
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
