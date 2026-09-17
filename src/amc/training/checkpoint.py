"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(payload: Dict[str, Any], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    # Load only trusted project checkpoints; these include metadata beyond tensors.
    return torch.load(path, map_location=map_location, weights_only=False)
