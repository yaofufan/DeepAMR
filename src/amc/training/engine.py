"""Training and evaluation loops."""

from __future__ import annotations

import time
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from amc.config import ExperimentConfig, as_dict, config_from_dict, resolve_src_path, save_config
from amc.evaluation.metrics import (
    classification_metrics,
    confusion_matrix,
    per_class_snr_accuracy,
    per_group_accuracy,
    per_snr_accuracy,
)
from .checkpoint import load_checkpoint, save_checkpoint


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _unpack_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    x = batch["x"].to(device, non_blocking=True).float()
    y = batch["y"].to(device, non_blocking=True).long()
    snr = batch["snr"].cpu().numpy()
    channel = batch.get("channel")
    channel_np = (
        channel.cpu().numpy()
        if channel is not None
        else np.zeros(y.numel(), dtype=np.int64)
    )
    return x, y, snr, channel_np


def _make_scheduler(optimizer: torch.optim.Optimizer, config: ExperimentConfig):
    name = config.train.scheduler.lower()
    if name in {"", "none", "off"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.epochs,
            eta_min=config.train.min_lr,
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.train.scheduler_factor,
            patience=config.train.scheduler_patience,
            threshold=0.0,
            threshold_mode="abs",
            min_lr=config.train.min_lr,
        )
    raise ValueError(f"Unknown LR scheduler: {config.train.scheduler}")


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[list[str]] = None,
    channel_names: Optional[list[str]] = None,
    feature_layer: Optional[nn.Module] = None,
    feature_snr: Optional[int] = None,
    export_predictions: bool = False,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    all_true, all_pred, all_snr, all_channel = [], [], [], []
    feature_batches, feature_true, feature_channel = [], [], []
    captured_features: list[torch.Tensor] = []

    def capture_features(_module, inputs) -> None:
        captured_features.append(inputs[0].detach().cpu())

    hook = (
        feature_layer.register_forward_pre_hook(capture_features)
        if feature_layer is not None
        else None
    )
    try:
        for batch in loader:
            x, y, snr, channel = _unpack_batch(batch, device)
            captured_features.clear()
            logits = model(x)
            if criterion is not None:
                total_loss += float(criterion(logits, y).item()) * y.numel()
            sample_count += y.numel()
            pred = torch.argmax(logits, dim=1)
            y_np = y.cpu().numpy()
            all_true.append(y_np)
            all_pred.append(pred.cpu().numpy())
            all_snr.append(snr)
            all_channel.append(channel)

            if feature_layer is not None:
                if len(captured_features) != 1:
                    raise RuntimeError("Expected one feature tensor from the selected layer.")
                mask = np.ones(len(snr), dtype=bool) if feature_snr is None else snr == feature_snr
                if np.any(mask):
                    feature_batches.append(captured_features[0].numpy()[mask])
                    feature_true.append(y_np[mask])
                    feature_channel.append(channel[mask])
    finally:
        if hook is not None:
            hook.remove()

    if sample_count == 0:
        raise ValueError("Evaluation loader contains no samples.")
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    snr_values = np.concatenate(all_snr)
    channel_values = np.concatenate(all_channel)
    metrics = classification_metrics(y_true, y_pred, num_classes)
    metrics["loss"] = total_loss / sample_count
    metrics["per_snr_accuracy"] = per_snr_accuracy(y_true, y_pred, snr_values)
    metrics["per_modulation_snr_accuracy"] = per_class_snr_accuracy(
        y_true,
        y_pred,
        snr_values,
        class_names,
    )
    metrics["per_channel_accuracy"] = per_group_accuracy(
        y_true,
        y_pred,
        channel_values,
        channel_names,
    )
    metrics["per_channel_snr_accuracy"] = {
        (channel_names[int(c)] if channel_names is not None else str(c)): per_snr_accuracy(
            y_true[channel_values == c], y_pred[channel_values == c], snr_values[channel_values == c]
        )
        for c in np.unique(channel_values)
    }
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, num_classes)
    if export_predictions:
        metrics["_prediction_export"] = {
            "y_true": y_true, "y_pred": y_pred, "snr": snr_values, "channel": channel_values,
        }
    if feature_layer is not None:
        if not feature_batches:
            raise ValueError(f"No evaluation samples found at SNR {feature_snr}dB.")
        metrics["_feature_export"] = {
            "features": np.concatenate(feature_batches),
            "labels": np.concatenate(feature_true),
            "channel": np.concatenate(feature_channel),
        }
    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    class_names: List[str],
    channel_names: Optional[List[str]] = None,
    resume: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    device = resolve_device(config.train.device)
    model = model.to(device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    criterion = nn.CrossEntropyLoss()
    if config.train.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    amp_enabled = config.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scheduler = _make_scheduler(optimizer, config)
    scheduler_name = config.train.scheduler if scheduler is not None else "none"
    print(
        f"Using device: {device} ({device_name}); amp={amp_enabled}; "
        f"scheduler={scheduler_name}; lr={_current_lr(optimizer):.6g}; "
        f"loss=ce; parameters={sum(p.numel() for p in model.parameters()):,}"
    )

    output_dir = resolve_src_path(config.train.output_dir)

    best_loss = float("inf")
    best_acc = -1.0
    bad_epochs = 0
    history: List[Dict[str, float]] = []
    start_epoch = 1
    if resume is not None:
        resume_path = resolve_src_path(resume)
        saved = load_checkpoint(str(resume_path), map_location="cpu")
        required = {"optimizer", "scheduler", "scaler", "rng", "best_loss", "bad_epochs"}
        if not required.issubset(saved):
            raise ValueError("This checkpoint lacks complete training state and cannot be resumed.")
        current = as_dict(config)
        saved_config = as_dict(config_from_dict(saved["config"]))
        if saved_config["model"] != current["model"]:
            raise ValueError("Resume model configuration does not match the checkpoint.")
        for key in ("scheduler", "scheduler_patience", "scheduler_factor", "min_lr",
                    "accumulation_steps", "amp", "grad_clip_norm", "early_stop_patience",
                    "lr", "weight_decay", "seed"):
            if saved_config["train"].get(key) != current["train"].get(key):
                raise ValueError(f"Resume training setting changed: {key}")
        for key, value in saved_config["data"].items():
            if key not in {"num_workers", "prefetch_factor", "persistent_workers"} and current["data"].get(key) != value:
                raise ValueError(f"Resume data setting changed: {key}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None and saved["scheduler"] is not None:
            scheduler.load_state_dict(saved["scheduler"])
        if amp_enabled and saved["scaler"]:
            scaler.load_state_dict(saved["scaler"])
        best_loss, best_acc = saved["best_loss"], saved["best_acc"]
        bad_epochs, history = saved["bad_epochs"], saved["history"]
        start_epoch = saved["epoch"] + 1
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        if device.type == "cuda" and saved["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        sampler = train_loader.batch_sampler
        if hasattr(sampler, "epoch"):
            sampler.epoch = saved.get("sampler_epoch", start_epoch - 1)
        print(f"Resuming at epoch {start_epoch} from {resume_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.json")
    print(f"Run directory: {output_dir.resolve()}")

    for epoch in range(start_epoch, config.train.epochs + 1):
        start = time.time()
        model.train()
        running_loss = 0.0
        accumulation = config.train.accumulation_steps
        accumulated_samples = 0
        seen = 0
        optimizer_updated = False
        for step, batch in enumerate(train_loader, start=1):
            x, y, _, _ = _unpack_batch(batch, device)
            if (step - 1) % accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
                accumulated_samples = 0
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite training loss at epoch={epoch}, step={step}.")
            batch_size = y.numel()
            accumulated_samples += batch_size
            scaler.scale(loss if accumulation == 1 else loss * batch_size).backward()
            if step % accumulation == 0 or step == len(train_loader):
                if config.train.grad_clip_norm is not None or accumulation > 1:
                    scaler.unscale_(optimizer)
                if accumulation > 1:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated_samples)
                if config.train.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_updated |= scaler.get_scale() >= previous_scale

            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            if config.train.log_interval and step % config.train.log_interval == 0:
                print(
                    f"epoch={epoch} step={step} "
                    f"train_loss={running_loss / max(1, seen):.4f}"
                )

        if seen == 0:
            raise ValueError("Training loader contains no samples.")
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion=criterion,
            num_classes=len(class_names),
            class_names=class_names,
            channel_names=channel_names,
        )
        train_loss = running_loss / max(1, seen)
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "lr": _current_lr(optimizer),
            "seconds": float(time.time() - start),
        }
        history.append(row)
        print(
            "epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            "val_acc={val_accuracy:.4f} val_f1={val_macro_f1:.4f} "
            "lr={lr:.6g} seconds={seconds:.1f}".format(
                **row
            )
        )

        improved = float(val_metrics["loss"]) < best_loss
        if improved:
            best_loss = float(val_metrics["loss"])
            best_acc = float(val_metrics["accuracy"])
            bad_epochs = 0
        else:
            bad_epochs += 1
        should_stop = not improved and bad_epochs >= config.train.early_stop_patience

        if scheduler is not None and optimizer_updated and not should_stop:
            if config.train.scheduler.lower() == "plateau":
                scheduler.step(float(val_metrics["loss"]))
            else:
                scheduler.step()

        payload = {
            "model": model.state_dict(), "config": as_dict(config),
            "class_names": class_names, "channel_names": channel_names,
            "epoch": epoch, "metrics": val_metrics, "history": history,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(), "best_loss": best_loss, "best_acc": best_acc,
            "bad_epochs": bad_epochs, "sampler_epoch": getattr(train_loader.batch_sampler, "epoch", epoch),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
        }
        save_checkpoint(payload, str(output_dir / "last.pt"))
        if improved:
            save_checkpoint(payload, str(output_dir / "best.pt"))
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if should_stop:
            print(f"Early stopping at epoch {epoch}: validation loss did not improve.")
            break

    return model, {
        "best_val_loss": best_loss,
        "accuracy_at_best_val_loss": best_acc,
        "history": history,
    }
