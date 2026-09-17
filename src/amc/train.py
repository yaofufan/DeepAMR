"""Train an AMR model from a config file."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Support both ``python -m amc.train`` and direct IDE execution of this file.
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from amc.config import load_config
from amc.data import build_dataloaders
from amc.models import build_model
from amc.training import train
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "resnet.json"))
    parser.add_argument("--resume", default=None, help="Trusted last.pt with complete training state.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = config.data.seed if config.train.seed is None else config.train.seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    data = build_dataloaders(config.data)
    config.model.num_classes = len(data.class_names)
    config.model.input_length = config.data.input_length
    model = build_model(config.model)
    train(model, data.train, data.val, config, data.class_names, data.channel_names, resume=args.resume)


if __name__ == "__main__":
    main()
