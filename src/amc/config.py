"""Configuration helpers for experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


SRC_ROOT = Path(__file__).resolve().parents[1]


def resolve_src_path(path: Union[str, Path]) -> Path:
    """Resolve project-relative paths against src, independent of the shell cwd."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else SRC_ROOT / path


@dataclass
class DataConfig:
    dataset: str = "matlab"
    path: Optional[str] = None
    mat_x_key: str = "X"
    mat_y_key: str = "y"
    mat_snr_key: Optional[str] = "snr"
    mat_channel_key: Optional[str] = "channel"
    mat_split_key: Optional[str] = "split"
    input_length: int = 512
    class_names: Optional[List[str]] = None
    channel_names: Optional[List[str]] = None
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    batch_size: int = 1024
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    hdf5_block_size: int = 64
    normalize: str = "rms"
    seed: int = 42


@dataclass
class ModelConfig:
    name: str = "oshea_resnet"
    num_classes: int = 15
    input_channels: int = 2
    input_length: int = 512
    hidden_channels: int = 32
    dropout: float = 0.2
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    device: str = "auto"
    output_dir: str = "runs/default"
    early_stop_patience: int = 15
    grad_clip_norm: Optional[float] = 5.0
    amp: bool = False
    scheduler: str = "plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    log_interval: int = 50
    accumulation_steps: int = 1
    seed: Optional[int] = None


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def as_dict(config: ExperimentConfig) -> Dict[str, Any]:
    return {
        "data": config.data.__dict__,
        "model": {
            key: value
            for key, value in config.model.__dict__.items()
        },
        "train": config.train.__dict__,
    }


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load a JSON or YAML config file.

    YAML support is optional and only used when PyYAML is installed.
    """

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install PyYAML or use a JSON config file.") from exc
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)

    return config_from_dict(raw)


def config_from_dict(raw: Dict[str, Any]) -> ExperimentConfig:
    """Load current settings and harmless defaults in older baseline runs."""
    if raw.get("loss", {}).get("name", "ce") != "ce":
        raise ValueError("Only cross-entropy baseline training is supported.")
    merged = _deep_update(as_dict(ExperimentConfig()), raw)
    data = dict(merged["data"])
    if data.pop("train_sampler", "blocks") != "blocks":
        raise ValueError("Only the baseline data sampler is supported.")
    data.pop("pair_block_size", None)
    return ExperimentConfig(
        data=DataConfig(**data),
        model=ModelConfig(**merged["model"]),
        train=TrainConfig(**merged["train"]),
    )


def save_config(config: ExperimentConfig, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_dict(config), indent=2), encoding="utf-8")
