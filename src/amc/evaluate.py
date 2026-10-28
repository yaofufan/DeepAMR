"""Evaluate a trained AMR checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

# Support both ``python -m amc.evaluate`` and direct IDE execution of this file.
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from amc.config import ExperimentConfig, load_config, resolve_src_path
from amc.data import build_dataloaders
from amc.models import build_model
from amc.training.checkpoint import load_checkpoint
from amc.training.engine import evaluate, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "ours_saftformer" / "full.json"))
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to evaluate; defaults to best.pt in the selected config's training output directory.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", default=None,
                        help="Evaluation directory; defaults to eval_<training directory> beside the selected config's training output.")
    parser.add_argument("--feature-snr", type=int, default=0)
    parser.add_argument("--feature-samples-per-class", type=int, default=300)
    parser.add_argument("--skip-feature-export", action="store_true")
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Override evaluation loader workers without changing the saved config; 0 disables multiprocessing.",
    )
    args = parser.parse_args()
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be non-negative.")
    if args.checkpoint is None or args.output_dir is None:
        training_dir = Path(load_config(args.config).train.output_dir)
        if args.checkpoint is None:
            args.checkpoint = str(resolve_src_path(training_dir) / "best.pt")
        if args.output_dir is None:
            args.output_dir = (training_dir.parent / f"eval_{training_dir.name}").as_posix()
    return args


def final_linear(model: nn.Module) -> nn.Module:
    feature_layer = getattr(model, "feature_layer", None)
    if isinstance(feature_layer, nn.Module):
        return feature_layer
    search_root = model.classifier if hasattr(model, "classifier") else model
    layers = [layer for layer in search_root.modules() if isinstance(layer, nn.Linear)]
    if not layers:
        raise ValueError("The model has no Linear classification layer to capture.")
    return layers[-1]


def balanced_feature_indices(
    labels: np.ndarray,
    channels: np.ndarray,
    num_classes: int,
    samples_per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in range(num_classes):
        class_mask = labels == class_id
        channel_ids = np.unique(channels[class_mask])
        if channel_ids.size == 0:
            raise ValueError(f"No feature samples found for class {class_id}.")
        base, remainder = divmod(samples_per_class, len(channel_ids))
        class_parts: list[np.ndarray] = []
        for position, channel_id in enumerate(channel_ids):
            candidates = np.flatnonzero(class_mask & (channels == channel_id))
            take = base + int(position < remainder)
            if len(candidates) < take:
                raise ValueError(
                    f"Class {class_id}, channel {channel_id} has {len(candidates)} "
                    f"samples, fewer than the requested {take}."
                )
            class_parts.append(rng.choice(candidates, size=take, replace=False))
        selected.append(np.concatenate(class_parts))
    return np.concatenate(selected)


def write_feature_results(
    output_dir: Path,
    feature_data: dict,
    class_names: list[str],
    channel_names: list[str],
    feature_snr: int,
    samples_per_class: int,
    split: str,
    model_name: str,
    seed: int,
) -> Path:
    indices = balanced_feature_indices(
        feature_data["labels"],
        feature_data["channel"],
        len(class_names),
        samples_per_class,
        seed,
    )
    output_path = output_dir / f"tsne_features_{feature_snr}dB.mat"
    savemat(
        output_path,
        {
            "features": feature_data["features"][indices].astype(np.float32),
            "labels": feature_data["labels"][indices].astype(np.int64),
            "snr": np.full(len(indices), feature_snr, dtype=np.int64),
            "channel": feature_data["channel"][indices].astype(np.int64),
            "classNames": np.asarray(class_names, dtype=object),
            "channelNames": np.asarray(channel_names, dtype=object),
            "targetSnr": np.asarray([[feature_snr]], dtype=np.int64),
            "splitName": np.asarray([split], dtype=object),
            "modelName": np.asarray([model_name], dtype=object),
        },
        do_compression=True,
    )
    return output_path


def write_csv_results(
    output_dir: Path,
    metrics: dict,
    class_names: list[str],
    channel_names: list[str],
) -> None:
    with (output_dir / "snr_accuracy.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["snr", "accuracy"])
        writer.writerows(sorted(metrics["per_snr_accuracy"].items()))

    with (output_dir / "modulation_snr_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["modulation", "snr", "accuracy"])
        curves = metrics["per_modulation_snr_accuracy"]
        for name in class_names:
            writer.writerows(
                [name, snr, accuracy]
                for snr, accuracy in sorted(curves[name].items())
            )

    with (output_dir / "channel_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["channel", "accuracy"])
        channel_accuracy = metrics["per_channel_accuracy"]
        writer.writerows([name, channel_accuracy[name]] for name in channel_names)

    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted", *class_names])
        for name, row in zip(class_names, metrics["confusion_matrix"]):
            writer.writerow([name, *np.asarray(row, dtype=np.int64).tolist()])

    with (output_dir / "channel_snr_accuracy.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["channel", "snr", "accuracy"])
        for channel, curve in metrics["per_channel_snr_accuracy"].items():
            writer.writerows([channel, snr, accuracy] for snr, accuracy in sorted(curve.items()))


def main() -> None:
    args = parse_args()
    config: ExperimentConfig = load_config(args.config)
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    bundle = build_dataloaders(config.data)
    config.model.num_classes = len(bundle.class_names)
    config.model.input_length = config.data.input_length
    model = build_model(config.model)
    checkpoint_path = resolve_src_path(args.checkpoint)
    ckpt = load_checkpoint(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    checkpoint_hash = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            checkpoint_hash.update(block)
    device = resolve_device(config.train.device)
    feature_layer = None if args.skip_feature_export else final_linear(model)
    evaluation_start = time.perf_counter()
    metrics = evaluate(
        model.to(device),
        getattr(bundle, args.split),
        device,
        criterion=nn.CrossEntropyLoss(),
        num_classes=len(bundle.class_names),
        class_names=bundle.class_names,
        channel_names=bundle.channel_names,
        feature_layer=feature_layer,
        feature_snr=args.feature_snr,
        export_predictions=True,
    )
    elapsed_seconds = time.perf_counter() - evaluation_start

    output_dir = resolve_src_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_data = metrics.pop("_feature_export", None)
    predictions = metrics.pop("_prediction_export")
    savemat(output_dir / "predictions.mat", {
        **predictions,
        "classNames": np.asarray(bundle.class_names, dtype=object),
        "channelNames": np.asarray(bundle.channel_names, dtype=object),
    }, do_compression=True)
    serializable = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metrics.items()
    }
    serializable["split"] = args.split
    serializable["model"] = config.model.name
    serializable["model_display_name"] = getattr(model, "model_name", config.model.name)
    serializable["model_full_name"] = getattr(model, "model_full_name", config.model.name)
    serializable["class_names"] = bundle.class_names
    serializable["channel_names"] = bundle.channel_names
    (output_dir / "metrics.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    write_csv_results(
        output_dir,
        metrics,
        bundle.class_names,
        bundle.channel_names,
    )
    if feature_data is not None:
        feature_path = write_feature_results(
            output_dir,
            feature_data,
            bundle.class_names,
            bundle.channel_names,
            args.feature_snr,
            args.feature_samples_per_class,
            args.split,
            config.model.name,
            config.data.seed,
        )
        print(f"Saved t-SNE source features to {feature_path.resolve()}")
    provenance = {
        "config_file": str(Path(args.config).resolve()),
        "checkpoint_file": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash.hexdigest(),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_selection": checkpoint_path.name,
        "split": args.split,
        "num_workers_override": args.num_workers,
        "precision": "float32",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "evaluated_samples": len(predictions["y_true"]),
        "test_samples": len(predictions["y_true"]) if args.split == "test" else 0,
        "model": config.model.name,
        "elapsed_seconds": elapsed_seconds,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "evaluation_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    summary = {
        "accuracy": serializable["accuracy"],
        "macro_f1": serializable["macro_f1"],
        "kappa": serializable["kappa"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved raw evaluation results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
