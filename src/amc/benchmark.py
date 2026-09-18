"""Summarize AMR accuracy and model-complexity results."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

# Support both ``python -m amc.benchmark`` and direct IDE execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from amc.config import ExperimentConfig, load_config, resolve_src_path
from amc.models import build_model
from amc.training.checkpoint import load_checkpoint
from amc.training.engine import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute accuracy summary, parameters, MACs, and latency."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "resnet.json"))
    parser.add_argument(
        "--checkpoint", default=str(ROOT / "runs" / "oshea_resnet" / "best.pt")
    )
    parser.add_argument(
        "--metrics", default=str(ROOT / "runs" / "eval_oshea_resnet" / "metrics.json")
    )
    parser.add_argument(
        "--output", default=str(ROOT / "runs" / "eval_oshea_resnet" / "benchmark.json")
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Latency device. Defaults to train.device from the config.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument(
        "--amp", action="store_true", help="Measure CUDA latency with autocast FP16."
    )
    return parser.parse_args()


def load_trained_model(config: ExperimentConfig, checkpoint_path: Path):
    checkpoint = load_checkpoint(str(checkpoint_path), map_location="cpu")
    class_names = checkpoint.get("class_names")
    if class_names:
        config.model.num_classes = len(class_names)
    config.model.input_length = config.data.input_length
    model = build_model(config.model)
    model.load_state_dict(checkpoint["model"])
    return model


def summarize_accuracy(metrics_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    curve = metrics.get("per_snr_accuracy")
    if not curve:
        raise ValueError(f"per_snr_accuracy is missing or empty in {metrics_path}.")

    snr_accuracy = {int(snr): float(accuracy) for snr, accuracy in curve.items()}
    max_accuracy = max(snr_accuracy.values())
    best_snrs = sorted(
        snr for snr, accuracy in snr_accuracy.items() if np.isclose(accuracy, max_accuracy)
    )
    average_accuracy = float(
        metrics.get("accuracy", np.mean(list(snr_accuracy.values())))
    )
    return {
        "average_accuracy": average_accuracy,
        "mean_snr_accuracy": float(np.mean(list(snr_accuracy.values()))),
        "max_accuracy": max_accuracy,
        "best_snr_db": best_snrs,
        "num_snr_points": len(snr_accuracy),
        "split": metrics.get("split", "unknown"),
    }


def count_macs(model: torch.nn.Module, input_shape: tuple[int, int, int]) -> float:
    try:
        from thop import profile
    except ImportError as exc:
        raise RuntimeError(
            "Install THOP to calculate MACs: python -m pip install thop"
        ) from exc

    profile_model = copy.deepcopy(model).cpu().eval()
    sample = torch.zeros(input_shape, dtype=torch.float32)
    with torch.inference_mode():
        macs, _ = profile(profile_model, inputs=(sample,), verbose=False)
    return float(macs)


def measure_latency(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int],
    device: torch.device,
    warmup: int,
    repetitions: int,
    amp: bool,
) -> float:
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be non-negative and repetitions must be positive.")
    if amp and device.type != "cuda":
        raise ValueError("--amp latency measurement requires a CUDA device.")

    model = model.to(device).eval()
    sample = torch.randn(input_shape, device=device)
    autocast_enabled = amp and device.type == "cuda"
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repetitions):
                with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
                    model(sample)
            end.record()
            torch.cuda.synchronize(device)
            return float(start.elapsed_time(end) / repetitions)

        start_time = time.perf_counter()
        for _ in range(repetitions):
            model(sample)
        return float((time.perf_counter() - start_time) * 1000.0 / repetitions)


def write_results(output_path: Path, results: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    row = {
        "model": results["model"],
        "ave_acc_pct": results["average_accuracy_pct"],
        "max_acc_pct": results["max_accuracy_pct"],
        "params_k": results["params_k"],
        "macs_m": results["macs_m"],
        "latency_ms": results["latency_ms"],
    }
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")

    config = load_config(args.config)
    checkpoint_path = resolve_src_path(args.checkpoint)
    metrics_path = resolve_src_path(args.metrics)
    output_path = resolve_src_path(args.output)
    model = load_trained_model(config, checkpoint_path)

    accuracy = summarize_accuracy(metrics_path)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    latency_input_shape = (
        args.batch_size,
        config.model.input_channels,
        config.model.input_length,
    )
    macs_input_shape = (
        1,
        config.model.input_channels,
        config.model.input_length,
    )
    macs = count_macs(model, macs_input_shape)
    requested_device = config.train.device if args.device is None else args.device
    device = resolve_device(requested_device)
    latency_per_batch = measure_latency(
        model,
        latency_input_shape,
        device,
        args.warmup,
        args.repetitions,
        args.amp,
    )

    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    results = {
        "model": config.model.name,
        "average_accuracy": accuracy["average_accuracy"],
        "average_accuracy_pct": 100.0 * accuracy["average_accuracy"],
        "mean_snr_accuracy": accuracy["mean_snr_accuracy"],
        "mean_snr_accuracy_pct": 100.0 * accuracy["mean_snr_accuracy"],
        "max_accuracy": accuracy["max_accuracy"],
        "max_accuracy_pct": 100.0 * accuracy["max_accuracy"],
        "best_snr_db": accuracy["best_snr_db"],
        "num_snr_points": accuracy["num_snr_points"],
        "split": accuracy["split"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "params_k": total_params / 1_000.0,
        "macs": macs,
        "macs_m": macs / 1_000_000.0,
        "latency_ms": latency_per_batch,
        "latency_per_sample_ms": latency_per_batch / args.batch_size,
        "batch_latency_ms": latency_per_batch,
        "batch_size": args.batch_size,
        "input_shape": list(latency_input_shape),
        "macs_input_shape": list(macs_input_shape),
        "device": str(device),
        "device_name": device_name,
        "precision": "amp_fp16" if args.amp else "fp32",
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "metrics_file": str(metrics_path.resolve()),
        "checkpoint_file": str(checkpoint_path.resolve()),
    }
    write_results(output_path, results)
    print(json.dumps(results, indent=2))
    print(f"Saved benchmark JSON to {output_path.resolve()}")
    print(f"Saved paper-table CSV to {output_path.with_suffix('.csv').resolve()}")


if __name__ == "__main__":
    main()
