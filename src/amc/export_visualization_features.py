"""Export balanced test-only feature subsets without re-running full evaluation.

Run from src with ``python -m amc.export_visualization_features``. The exported
features are the input to the final classification Linear layer, captured with
the same pre-hook as amc.evaluate. Existing evaluation outputs are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader

from amc.config import load_config, resolve_src_path
from amc.data.datamodule import H5MatSignalDataset
from amc.data.matlab import load_matlab_hdf5_metadata
from amc.evaluate import balanced_feature_indices, final_linear
from amc.models import build_model
from amc.training.checkpoint import load_checkpoint
from amc.training.engine import resolve_device


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_overall(
    test_indices: np.ndarray,
    labels: np.ndarray,
    snr: np.ndarray,
    channels: np.ndarray,
    num_classes: int,
    samples_per_class: int,
    seed: int,
) -> np.ndarray:
    """Balance classes and SNR exactly, rotating extra channel allocations."""
    rng = np.random.default_rng(seed)
    levels = np.unique(snr[test_indices])
    channel_ids = np.unique(channels[test_indices])
    per_level, extra_levels = divmod(samples_per_class, len(levels))
    result: list[np.ndarray] = []
    for class_id in range(num_classes):
        class_indices = test_indices[labels[test_indices] == class_id]
        for level_position, level in enumerate(levels):
            level_indices = class_indices[snr[class_indices] == level]
            take = per_level + int(level_position < extra_levels)
            per_channel, extra_channels = divmod(take, len(channel_ids))
            start = (class_id * extra_channels + level_position * extra_channels) % len(channel_ids)
            extra = {(start + k) % len(channel_ids) for k in range(extra_channels)}
            for channel_position, channel_id in enumerate(channel_ids):
                candidates = level_indices[channels[level_indices] == channel_id]
                count = per_channel + int(channel_position in extra)
                if len(candidates) < count:
                    raise ValueError("Insufficient test examples in a class/SNR/channel cell.")
                result.append(rng.choice(candidates, size=count, replace=False))
    return np.concatenate(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ours_saftformer/full.json")
    parser.add_argument("--checkpoint", default="runs/saftformer_full/best.pt")
    parser.add_argument("--evaluation-dir", default="runs/eval_saftformer_full")
    parser.add_argument("--output-dir", default="result_plotting_matlab/feature_data")
    parser.add_argument("--snr-levels", type=int, nargs=2, default=[0, 10])
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.samples_per_class <= 0 or args.batch_size <= 0:
        parser.error("Sample and batch sizes must be positive.")
    if len(set(args.snr_levels)) != 2:
        parser.error("Choose two different SNR levels.")

    config_path = resolve_src_path(args.config).resolve()
    checkpoint_path = resolve_src_path(args.checkpoint).resolve()
    evaluation_dir = resolve_src_path(args.evaluation_dir).resolve()
    output_dir = resolve_src_path(args.output_dir).resolve()
    config = load_config(config_path)
    dataset_path = resolve_src_path(config.data.path).resolve()
    provenance_path = evaluation_dir / "evaluation_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != provenance["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the formal evaluation checkpoint.")
    if provenance.get("checkpoint_selection") != "best.pt" or provenance.get("split") != "test":
        raise ValueError("Expected the formal best.pt test evaluation.")

    y, snr, class_names, metadata = load_matlab_hdf5_metadata(
        dataset_path,
        y_key=config.data.mat_y_key,
        snr_key=config.data.mat_snr_key,
        channel_key=config.data.mat_channel_key,
        split_key=config.data.mat_split_key,
        class_names=config.data.class_names,
        channel_names=config.data.channel_names,
    )
    channels = metadata["channel"]
    split = metadata["split"]
    if split is None:
        raise ValueError("Stored test split labels are required for this export.")
    test_indices = np.flatnonzero(split == 2)
    reference = loadmat(evaluation_dir / "predictions.mat")
    for actual, field in [(y, "y_true"), (snr, "snr"), (channels, "channel")]:
        if not np.array_equal(actual[test_indices], reference[field].ravel()):
            raise ValueError(f"Stored predictions do not follow the ascending test-source order: {field}.")
    saved_predictions = reference["y_pred"].ravel()
    groups = {
        "overall": select_overall(
            test_indices, y, snr, channels, len(class_names), args.samples_per_class, args.seed
        )
    }
    for level in args.snr_levels:
        candidates = test_indices[snr[test_indices] == level]
        positions = balanced_feature_indices(
            y[candidates], channels[candidates], len(class_names), args.samples_per_class, args.seed
        )
        groups[f"{level}dB"] = candidates[positions]
    unique_indices = np.unique(np.concatenate(list(groups.values())))
    if np.any(split[unique_indices] != 2):
        raise AssertionError("A selected sample is outside the test split.")

    dataset = H5MatSignalDataset(
        str(dataset_path), config.data.mat_x_key, y, snr, channels, config.data.normalize
    )
    batches = [unique_indices[start:start + args.batch_size].tolist()
               for start in range(0, len(unique_indices), args.batch_size)]
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=0)
    checkpoint = load_checkpoint(str(checkpoint_path), map_location="cpu")
    config.model.num_classes = len(class_names)
    config.model.input_length = config.data.input_length
    model = build_model(config.model)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = resolve_device(args.device)
    model.to(device).eval()
    captured: list[torch.Tensor] = []

    def capture(_module, inputs) -> None:
        captured.append(inputs[0].detach().cpu())

    layer = final_linear(model)
    hook = layer.register_forward_pre_hook(capture)
    feature_batches, prediction_batches = [], []
    print(f"Exporting {len(unique_indices)} unique test examples on {device}; FP32, no autocast.", flush=True)
    try:
        with torch.inference_mode():
            for batch_position, batch in enumerate(loader):
                captured.clear()
                logits = model(batch["x"].to(device=device, dtype=torch.float32))
                if len(captured) != 1 or captured[0].ndim != 2:
                    raise RuntimeError("Expected one batch of pre-classifier feature vectors.")
                feature_batches.append(captured[0].numpy())
                prediction_batches.append(logits.argmax(dim=1).cpu().numpy())
                if (batch_position + 1) % 10 == 0:
                    print(f"feature batches {batch_position + 1}/{len(batches)}", flush=True)
    finally:
        hook.remove()
        if dataset._h5 is not None:
            dataset._h5.close()
    all_features = np.concatenate(feature_batches).astype(np.float32)
    predictions = np.concatenate(prediction_batches)
    test_positions = np.searchsorted(test_indices, unique_indices)
    expected_predictions = saved_predictions[test_positions]
    mismatches = int(np.count_nonzero(predictions != expected_predictions))
    if mismatches:
        raise RuntimeError(
            f"Subset predictions differ from formal evaluation at {mismatches}/{len(predictions)} samples."
        )
    if not np.isfinite(all_features).all():
        raise RuntimeError("Exported features contain non-finite values.")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint.get("epoch", provenance["checkpoint_epoch"])),
        "dataset_file": str(dataset_path), "config_file": str(config_path),
        "evaluation_provenance_file": str(provenance_path),
        "features": "Input to the final classification Linear: final-LayerNorm CLS token.",
        "device": str(device), "precision": "float32", "normalization": config.data.normalize,
        "selection_seed": args.seed, "unique_examples_inferred": len(unique_indices),
        "feature_dimensions": int(all_features.shape[1]),
        "reference_predictions_checked": len(predictions), "reference_prediction_mismatches": mismatches,
        "groups": {},
    }
    for name, source_indices in groups.items():
        positions = np.searchsorted(unique_indices, source_indices)
        features = all_features[positions]
        class_counts = np.bincount(y[source_indices], minlength=len(class_names))
        if not np.all(class_counts == args.samples_per_class):
            raise AssertionError("Class balance was not preserved.")
        class_channel = np.array([
            np.bincount(channels[source_indices][y[source_indices] == c], minlength=len(metadata["channel_names"]))
            for c in range(len(class_names))
        ])
        if np.any(class_channel.max(1) - class_channel.min(1) > 1):
            raise AssertionError("Channel balance exceeds one example per class.")
        levels = np.unique(snr[source_indices])
        target = np.nan if name == "overall" else int(name[:-2])
        filename = f"saftformer_features_{name}.mat"
        payload = {
            "features": features, "labels": y[source_indices],
            "snr": snr[source_indices], "channel": channels[source_indices],
            "classNames": np.asarray(class_names, dtype=object),
            "channelNames": np.asarray(metadata["channel_names"], dtype=object),
            "targetSnr": np.asarray([[target]]), "snrLevels": levels,
            "conditionName": np.asarray(["Overall" if name == "overall" else f"{target} dB"], dtype=object),
            "splitName": np.asarray(["test"], dtype=object),
            "modelName": np.asarray(["SAFTFormer"], dtype=object),
            "sourceIndices": source_indices, "sourceIndices1Based": source_indices + 1,
            "sourceIndexBase": np.asarray([[0]], dtype=np.int64),
            "testPredictionIndices": np.searchsorted(test_indices, source_indices),
            "predictions": predictions[positions],
            "selectionSeed": np.asarray([[args.seed]], dtype=np.int64),
            "checkpointHash": np.asarray([checkpoint_hash], dtype=object),
            "checkpointEpoch": np.asarray([[report["checkpoint_epoch"]]], dtype=np.int64),
            "featureLayer": np.asarray([report["features"]], dtype=object),
            "normalization": np.asarray([config.data.normalize], dtype=object),
            "sourceDataset": np.asarray([str(dataset_path)], dtype=object),
            "sourceCheckpoint": np.asarray([str(checkpoint_path)], dtype=object),
        }
        savemat(output_dir / filename, payload, do_compression=True)
        detail = {
            "file": filename, "shape": list(features.shape),
            "class_counts": class_counts.tolist(), "class_channel_counts": class_channel.tolist(),
            "snr_levels": levels.tolist(),
            "class_snr_counts": [[int(np.sum((y[source_indices] == c) & (snr[source_indices] == s)))
                                  for s in levels] for c in range(len(class_names))],
        }
        if name == "0dB" and (evaluation_dir / "tsne_features_0dB.mat").is_file():
            old = loadmat(evaluation_dir / "tsne_features_0dB.mat")
            detail["legacy_0dB_selection_metadata_match"] = bool(
                np.array_equal(old["labels"].ravel(), y[source_indices])
                and np.array_equal(old["channel"].ravel(), channels[source_indices])
            )
            if old["features"].shape == features.shape:
                detail["legacy_0dB_max_feature_abs_difference"] = float(np.max(np.abs(old["features"] - features)))
        report["groups"][name] = detail
        print(f"Saved {output_dir / filename}: {features.shape}", flush=True)
    (output_dir / "saftformer_feature_export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verified all {len(predictions)} subset predictions against the formal evaluation.", flush=True)


if __name__ == "__main__":
    main()
