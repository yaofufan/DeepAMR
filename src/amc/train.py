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

from amc.config import load_config, resolve_src_path
from amc.data import build_dataloaders
from amc.models import build_model
from amc.training import train
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_config = ROOT / "configs" / "ours_saftformer" / "full.json"
    parser.add_argument("--config", default=str(default_config))
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume", default=None,
        help="Trusted last.pt with complete training state; the default Full config resumes its own output directory if present. Custom configs require an explicit checkpoint.",
    )
    resume.add_argument(
        "--no-resume", dest="resume", action="store_const", const=None,
        help="Start a fresh training run instead of resuming the default checkpoint.",
    )
    parser.add_argument(
        "--next-stage", action="store_true", default=False,
        help="SE-TFA: load the resumed stage's best weights and start the next stage.",
    )
    args = parser.parse_args()
    explicit_resume_choice = any(
        arg in {"--resume", "--no-resume"} or arg.startswith("--resume=")
        for arg in sys.argv[1:]
    )
    # Derive the checkpoint from this Full preset, never an older architecture.
    if Path(args.config).resolve() == default_config.resolve() and not explicit_resume_choice:
        config = load_config(args.config)
        default_resume = resolve_src_path(config.train.output_dir) / "last.pt"
        if default_resume.is_file():
            args.resume = str(default_resume)
    if args.next_stage and args.resume is None:
        parser.error("--next-stage requires --resume.")
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.next_stage and config.train.protocol != "se_tfa_paper":
        raise ValueError("--next-stage is only supported for SE-TFA staged training.")
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
    if hasattr(model, "FULL_NAME"):
        print(f"{model.model_name}: {getattr(model, 'model_full_name', model.FULL_NAME)}")
    train(
        model,
        data.train,
        data.val,
        config,
        data.class_names,
        data.channel_names,
        resume=args.resume,
        next_stage=args.next_stage,
    )


if __name__ == "__main__":
    main()
