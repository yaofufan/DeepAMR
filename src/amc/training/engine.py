"""Training and evaluation loops."""

from __future__ import annotations

import time
import json
import random
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from amc.config import ExperimentConfig, as_dict, config_from_dict, resolve_src_path, save_config
from amc.data import filter_dataloader_by_snr
from amc.evaluation.metrics import (
    classification_metrics,
    confusion_matrix,
    per_class_snr_accuracy,
    per_group_accuracy,
    per_snr_accuracy,
)
from amc.models.baselines import SETFA
from .checkpoint import load_checkpoint, save_checkpoint


SE_TFA_STAGE_ORDER = ("classifier_pretrain", "enhancement", "joint_finetune")


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


def _make_scheduler(optimizer: torch.optim.Optimizer, config: ExperimentConfig,
                    steps_per_epoch: Optional[int] = None):
    name = config.train.scheduler.lower()
    if name in {"", "none", "off"}:
        return None
    if name == "step":
        settings = config.train
        if (not isinstance(settings.scheduler_step_size, int)
                or isinstance(settings.scheduler_step_size, bool)
                or settings.scheduler_step_size < 1
                or not math.isfinite(settings.scheduler_gamma)
                or not 0 < settings.scheduler_gamma <= 1):
            raise ValueError("StepLR requires positive integer step_size and gamma in (0,1].")
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=settings.scheduler_step_size, gamma=settings.scheduler_gamma)
    if name == "warmup_cosine":
        settings = config.train
        if (not isinstance(steps_per_epoch, int) or isinstance(steps_per_epoch, bool)
                or steps_per_epoch < 1):
            raise ValueError("warmup_cosine requires positive optimizer steps_per_epoch.")
        if (not isinstance(settings.epochs, int) or isinstance(settings.epochs, bool)
                or settings.epochs < 1 or not isinstance(settings.warmup_epochs, int)
                or isinstance(settings.warmup_epochs, bool)
                or not 0 <= settings.warmup_epochs < settings.epochs):
            raise ValueError("warmup_epochs must be an integer in [0, epochs).")
        if (not math.isfinite(settings.warmup_start_factor)
                or not 0 < settings.warmup_start_factor <= 1):
            raise ValueError("warmup_start_factor must lie in (0,1].")
        if (not math.isfinite(settings.lr) or settings.lr <= 0
                or not math.isfinite(settings.min_lr) or not 0 <= settings.min_lr <= settings.lr):
            raise ValueError("warmup_cosine requires 0 <= min_lr <= lr and lr > 0.")
        total_steps = settings.epochs * steps_per_epoch
        warmup_steps = settings.warmup_epochs * steps_per_epoch
        start_factor, minimum = settings.warmup_start_factor, settings.min_lr / settings.lr

        def lr_factor(step: int) -> float:
            if step < warmup_steps:
                return start_factor + (1 - start_factor) * step / warmup_steps
            progress = min(1.0, max(0.0, (step - warmup_steps) / (total_steps - warmup_steps)))
            return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
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


def _make_optimizer(model: nn.Module, config: ExperimentConfig):
    optimizers = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}
    name = config.train.optimizer.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {config.train.optimizer}")
    return optimizers[name](model.parameters(), lr=config.train.lr,
                            weight_decay=config.train.weight_decay)


def _snr_weighted_loss(logits: torch.Tensor, targets: torch.Tensor,
                       snr: torch.Tensor, m: float, j: float) -> torch.Tensor:
    """DHN equations (11)-(12): mean(sigmoid(m*(SNR+j)) * sample CE).

    Reference: https://doi.org/10.1109/ACCESS.2019.2928463
    Weighting is training-only; validation/test report ordinary cross entropy.
    """
    if not math.isfinite(m) or m <= 0 or not math.isfinite(j):
        raise ValueError("SNR weights require finite m > 0 and finite j.")
    snr = torch.as_tensor(snr, device=logits.device, dtype=torch.float32).reshape(-1)
    if snr.numel() != targets.numel() or not bool(torch.isfinite(snr).all()):
        raise ValueError("SNR weighting requires one finite SNR value per sample.")
    weights = torch.sigmoid(m * (snr + j))
    return (nn.functional.cross_entropy(logits, targets, reduction="none") * weights).mean()


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
                modulation_loss = criterion(logits, y)
                total_loss += float(modulation_loss.item()) * y.numel()
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
    # Track the previous failure mode without changing checkpoint selection.
    qam_recalls = {}
    for name in ("QAM16", "QAM64", "QAM256"):
        values = [float(accuracy) for level, accuracy in
                  metrics["per_modulation_snr_accuracy"].get(name, {}).items()
                  if int(level) >= 10]
        if values:
            qam_recalls[name] = float(np.mean(values))
    if len(qam_recalls) == 3:
        metrics["qam_high_macro_recall"] = float(np.mean(list(qam_recalls.values())))
        metrics["qam_high_recall"] = qam_recalls
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


def _train_standard(
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
    optimizer = _make_optimizer(model, config)
    amp_enabled = config.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scheduler = _make_scheduler(optimizer, config,
                                math.ceil(len(train_loader) / config.train.accumulation_steps))
    step_scheduler = config.train.scheduler.lower() == "warmup_cosine"
    scheduler_name = config.train.scheduler if scheduler is not None else "none"
    print(
        f"Using device: {device} ({device_name}); amp={amp_enabled}; "
        f"scheduler={scheduler_name}; lr={_current_lr(optimizer):.6g}; "
        f"optimizer={config.train.optimizer}; "
        f"loss={'snr_weighted_ce' if config.train.snr_weighting else 'ce'}; "
        f"parameters={sum(p.numel() for p in model.parameters()):,}"
    )

    output_dir = resolve_src_path(config.train.output_dir)

    best_loss = float("inf")
    best_acc = -1.0
    accuracy_at_best_loss = -1.0
    best_loss_epoch = best_acc_epoch = None
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
        if step_scheduler and saved_config["train"]["epochs"] != config.train.epochs:
            raise ValueError("Resume epochs changed the warmup_cosine schedule horizon.")
        for key in ("scheduler", "scheduler_patience", "scheduler_factor", "min_lr",
                    "warmup_epochs", "warmup_start_factor",
                    "accumulation_steps", "amp", "grad_clip_norm", "early_stop_patience",
                    "lr", "weight_decay", "seed", "protocol", "stages",
                    "optimizer", "scheduler_step_size", "scheduler_gamma",
                    "snr_weighting", "snr_weight_m", "snr_weight_j"):
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
        bad_epochs, history = saved["bad_epochs"], saved["history"]
        best_loss = float(saved["best_loss"])
        loss_record = min(history, key=lambda row: row["val_loss"], default=None)
        acc_record = max(history, key=lambda row: row["val_accuracy"], default=None)
        # Legacy checkpoints stored accuracy at minimum CE in best_acc. The
        # history restores the actual accuracy maximum without inventing the
        # weights from that historical epoch.
        best_acc = max(float(saved.get("best_acc", -1.0)),
                       float(acc_record["val_accuracy"]) if acc_record else -1.0)
        accuracy_at_best_loss = float(saved.get(
            "accuracy_at_best_val_loss",
            loss_record["val_accuracy"] if loss_record else saved.get("best_acc", -1.0)))
        best_loss_epoch = saved.get(
            "best_loss_epoch", int(loss_record["epoch"]) if loss_record else None)
        best_acc_epoch = saved.get(
            "best_acc_epoch", int(acc_record["epoch"]) if acc_record else None)
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
        objective_total = 0.0
        optimizer_updated = False
        epoch_start_lr = last_used_lr = _current_lr(optimizer)
        for step, batch in enumerate(train_loader, start=1):
            x, y, _, _ = _unpack_batch(batch, device)
            if (step - 1) % accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
                accumulated_samples = 0
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)
                modulation_loss = criterion(logits, y)
                loss = modulation_loss
                if config.train.snr_weighting:
                    loss = _snr_weighted_loss(logits, y, batch["snr"],
                                              config.train.snr_weight_m,
                                              config.train.snr_weight_j)
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
                attempted_lr = _current_lr(optimizer)
                scaler.step(optimizer)
                scaler.update()
                updated = scaler.get_scale() >= previous_scale
                optimizer_updated |= updated
                if updated:
                    last_used_lr = attempted_lr
                    if step_scheduler:
                        scheduler.step()

            # Keep train_loss/val_loss as modulation CE for baseline comparison.
            running_loss += float(modulation_loss.item()) * batch_size
            if config.train.snr_weighting:
                objective_total += float(loss.item()) * batch_size
            seen += batch_size
            if config.train.log_interval and step % config.train.log_interval == 0:
                message = f"epoch={epoch} step={step} train_loss={running_loss / max(1, seen):.4f}"
                print(message)

        if seen == 0:
            raise ValueError("Training loader contains no samples.")
        train_seconds = time.time() - start
        validation_start = time.time()
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion=criterion,
            num_classes=len(class_names),
            class_names=class_names,
            channel_names=channel_names,
        )
        val_seconds = time.time() - validation_start
        train_loss = running_loss / max(1, seen)
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "lr": last_used_lr,
            "seconds": float(time.time() - start),
            "train_seconds": float(train_seconds),
            "val_seconds": float(val_seconds),
        }
        if step_scheduler:
            row.update(lr_start=epoch_start_lr, lr_next=_current_lr(optimizer))
        if config.train.snr_weighting:
            row["train_objective_loss"] = objective_total / seen
        if "qam_high_macro_recall" in val_metrics:
            row["val_qam_high_macro_recall"] = val_metrics["qam_high_macro_recall"]
            row.update({f"val_{name.lower()}_high_recall": value
                        for name, value in val_metrics["qam_high_recall"].items()})
        history.append(row)
        print(
            "epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            "val_acc={val_accuracy:.4f} val_f1={val_macro_f1:.4f} "
            "lr={lr:.6g} seconds={seconds:.1f} "
            "train_s={train_seconds:.1f} val_s={val_seconds:.1f}".format(
                **row
            )
        )

        if config.train.snr_weighting:
            print(f"  train_snr_weighted_loss={row['train_objective_loss']:.6f}")
        improved_loss = float(val_metrics["loss"]) < best_loss
        improved_acc = float(val_metrics["accuracy"]) > best_acc
        if improved_loss:
            best_loss = float(val_metrics["loss"])
            accuracy_at_best_loss = float(val_metrics["accuracy"])
            best_loss_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        if improved_acc:
            best_acc = float(val_metrics["accuracy"])
            best_acc_epoch = epoch
        # Accuracy improvements do not reset CE-based early stopping or alter
        # the existing validation-loss scheduler.
        should_stop = not improved_loss and bad_epochs >= config.train.early_stop_patience

        if scheduler is not None and not step_scheduler and optimizer_updated and not should_stop:
            if config.train.scheduler.lower() == "plateau":
                scheduler.step(float(val_metrics["loss"]))
            else:
                scheduler.step()

        payload = {
            "model": model.state_dict(), "config": as_dict(config),
            "model_display_name": getattr(model, "model_name", config.model.name),
            "model_full_name": getattr(model, "model_full_name", config.model.name),
            "class_names": class_names, "channel_names": channel_names,
            "epoch": epoch, "metrics": val_metrics, "history": history,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(), "best_loss": best_loss, "best_acc": best_acc,
            "accuracy_at_best_val_loss": accuracy_at_best_loss,
            "best_loss_epoch": best_loss_epoch, "best_acc_epoch": best_acc_epoch,
            "bad_epochs": bad_epochs, "sampler_epoch": getattr(train_loader.batch_sampler, "epoch", epoch),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
        }
        save_checkpoint(payload, str(output_dir / "last.pt"))
        if improved_loss:
            save_checkpoint(payload, str(output_dir / "best_loss.pt"))
            # Keep the old filename for existing baseline evaluation commands.
            save_checkpoint(payload, str(output_dir / "best.pt"))
            print(f"  Saved best_loss.pt: val_loss={best_loss:.6f}, epoch={epoch}")
        if improved_acc:
            save_checkpoint(payload, str(output_dir / "best_acc.pt"))
            print(f"  Saved best_acc.pt: val_acc={best_acc:.6f}, epoch={epoch}")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if should_stop:
            print(f"Early stopping at epoch {epoch}: validation loss did not improve.")
            break

    return model, {
        "best_val_loss": best_loss,
        "best_val_accuracy": best_acc,
        "accuracy_at_best_val_loss": accuracy_at_best_loss,
        "best_loss_epoch": best_loss_epoch, "best_acc_epoch": best_acc_epoch,
        "history": history,
    }


def _make_se_tfa_optimizer(name: str, parameters, lr: float):
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    raise ValueError(f"Unsupported SE-TFA stage optimizer: {name}")


def _make_se_tfa_scheduler(optimizer, settings: dict):
    if settings.get("scheduler", "none") == "none":
        return None
    if settings["scheduler"] != "plateau":
        raise ValueError("SE-TFA stages support only plateau or no scheduler.")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(settings["scheduler_factor"]),
        patience=int(settings["scheduler_patience"]),
        threshold=0.0,
        threshold_mode="abs",
    )


@torch.no_grad()
def _evaluate_se_tfa_stage(
    model: SETFA,
    loader: DataLoader,
    device: torch.device,
    forward: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    count = 0
    for batch in loader:
        x, y, _, _ = _unpack_batch(batch, device)
        logits = forward(x)
        loss_sum += float(criterion(logits, y).item()) * y.numel()
        correct += int((logits.argmax(1) == y).sum().item())
        count += y.numel()
    if count == 0:
        raise ValueError("SE-TFA validation loader contains no samples.")
    return loss_sum / count, correct / count


def _rng_state(device: torch.device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def _restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _run_se_tfa_stage(
    model: SETFA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    stage_index: int,
    settings: dict,
    history: list[dict],
    output_dir: Path,
    class_names: list[str],
    channel_names: Optional[list[str]],
    resume: Optional[dict],
) -> dict:
    stage = SE_TFA_STAGE_ORDER[stage_index]
    model.set_training_stage(stage)
    best_name = "best.pt" if stage == "joint_finetune" else f"{stage}_best.pt"
    last_name = "last.pt" if stage == "joint_finetune" else f"{stage}_last.pt"
    device = next(model.parameters()).device

    # Older checkpoints also carry enough state to detect a completed stage.
    if resume is not None and (
        int(resume["epoch"]) >= int(settings["max_epochs"])
        or int(resume["bad_epochs"]) >= int(settings["early_stop_patience"])
    ):
        best = load_checkpoint(str(output_dir / best_name), map_location="cpu")
        model.load_state_dict(best["model"])
        history[:] = resume["history"]
        _restore_rng(resume["rng"], device)
        print(f"Skipping completed SE-TFA stage {stage}; loaded {best_name}.")
        return best

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = _make_se_tfa_optimizer(
        settings["optimizer"], parameters, float(settings["lr"])
    )
    scheduler = _make_se_tfa_scheduler(optimizer, settings)
    amp_enabled = config.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    forward = model.forward_classifier if stage == "classifier_pretrain" else model
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_acc = -1.0
    bad_epochs = 0
    start_epoch = 1

    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if scheduler is not None and resume["scheduler"] is not None:
            scheduler.load_state_dict(resume["scheduler"])
        if amp_enabled and resume.get("scaler"):
            scaler.load_state_dict(resume["scaler"])
        best_loss = float(resume["best_loss"])
        best_acc = float(resume["best_acc"])
        bad_epochs = int(resume["bad_epochs"])
        start_epoch = int(resume["epoch"]) + 1
        history[:] = resume["history"]
        _restore_rng(resume["rng"], device)

    for epoch in range(start_epoch, int(settings["max_epochs"]) + 1):
        started = time.time()
        model.train()
        running_loss = 0.0
        seen = 0
        for step, batch in enumerate(train_loader, start=1):
            x, y, _, _ = _unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = forward(x)
                loss = criterion(logits, y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite loss in {stage}, epoch={epoch}, step={step}."
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * y.numel()
            seen += y.numel()
            if config.train.log_interval and step % config.train.log_interval == 0:
                print(
                    f"stage={stage} epoch={epoch} step={step} "
                    f"train_loss={running_loss / seen:.4f}"
                )
        if seen == 0:
            raise ValueError(f"SE-TFA {stage} training loader contains no samples.")

        val_loss, val_accuracy = _evaluate_se_tfa_stage(
            model, val_loader, device, forward
        )
        row = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"stage={stage} epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"lr={row['lr']:.6g} seconds={row['seconds']:.1f}"
        )

        improved = val_loss < best_loss
        if improved:
            best_loss, best_acc, bad_epochs = val_loss, val_accuracy, 0
        else:
            bad_epochs += 1
        if scheduler is not None:
            scheduler.step(val_loss)

        payload = {
            "model": model.state_dict(),
            "config": as_dict(config),
            "class_names": class_names,
            "channel_names": channel_names,
            "stage": stage,
            "stage_index": stage_index,
            "epoch": epoch,
            "metrics": {"loss": val_loss, "accuracy": val_accuracy},
            "history": history,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "best_loss": best_loss,
            "best_acc": best_acc,
            "bad_epochs": bad_epochs,
            "rng": _rng_state(device),
        }
        save_checkpoint(payload, str(output_dir / last_name))
        if improved:
            save_checkpoint(payload, str(output_dir / best_name))
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if bad_epochs >= int(settings["early_stop_patience"]):
            print(f"Stopping {stage}: validation loss did not improve.")
            break

    best = load_checkpoint(str(output_dir / best_name), map_location="cpu")
    model.load_state_dict(best["model"])
    return best


def _train_se_tfa_paper(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    class_names: list[str],
    channel_names: Optional[list[str]] = None,
    resume: Optional[str] = None,
    next_stage: bool = False,
) -> Tuple[nn.Module, Dict[str, object]]:
    if next_stage and resume is None:
        raise ValueError("--next-stage requires --resume.")
    if not isinstance(model, SETFA):
        raise TypeError("The se_tfa_paper protocol requires an SETFA model.")
    missing = [stage for stage in SE_TFA_STAGE_ORDER if stage not in config.train.stages]
    if missing:
        raise ValueError(f"Missing SE-TFA stage settings: {missing}")

    device = resolve_device(config.train.device)
    model = model.to(device)
    output_dir = resolve_src_path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.json")
    print(f"Using paper-style SE-TFA training on {device}; run directory: {output_dir}")

    resume_data = None
    resume_stage = 0
    history: list[dict] = []
    if resume is not None:
        resume_data = load_checkpoint(str(resolve_src_path(resume)), map_location="cpu")
        saved_config = as_dict(config_from_dict(resume_data["config"]))
        current_config = as_dict(config)
        if saved_config["model"] != current_config["model"]:
            raise ValueError("Resume model configuration does not match the checkpoint.")
        if saved_config["train"]["stages"] != current_config["train"]["stages"]:
            raise ValueError("Resume stage settings do not match the checkpoint.")
        resume_stage = int(resume_data.get("stage_index", -1))
        if resume_stage not in range(len(SE_TFA_STAGE_ORDER)):
            raise ValueError("Resume checkpoint is not an SE-TFA staged checkpoint.")
        if next_stage:
            if resume_stage == len(SE_TFA_STAGE_ORDER) - 1:
                raise ValueError("The final SE-TFA stage has no next stage.")
            stage_name = SE_TFA_STAGE_ORDER[resume_stage]
            best_path = resolve_src_path(resume).parent / f"{stage_name}_best.pt"
            if not best_path.is_file():
                raise FileNotFoundError(f"The stage's best checkpoint is required: {best_path}")
            stage_best = load_checkpoint(str(best_path), map_location="cpu")
            if (stage_best.get("stage_index") != resume_stage
                    or stage_best["config"]["model"] != resume_data["config"]["model"]
                    or stage_best["config"]["train"]["stages"] != saved_config["train"]["stages"]
                    or int(stage_best["epoch"]) > int(resume_data["epoch"])):
                raise ValueError("Best checkpoint does not match the resumed stage.")
            model.load_state_dict(stage_best["model"])
            history[:] = resume_data["history"]
            _restore_rng(resume_data["rng"], device)
            resume_stage += 1
            resume_data = None

    if resume_stage == 0:
        highest_snr = int(config.train.stages["classifier_pretrain"].get("snr", 18))
        high_snr_train = filter_dataloader_by_snr(
            train_loader, highest_snr, config.data, shuffle=True
        )
        high_snr_val = filter_dataloader_by_snr(
            val_loader, highest_snr, config.data, shuffle=False
        )

    best = None
    for stage_index, stage in enumerate(SE_TFA_STAGE_ORDER):
        if stage_index < resume_stage:
            continue
        stage_train = high_snr_train if stage == "classifier_pretrain" else train_loader
        stage_val = high_snr_val if stage == "classifier_pretrain" else val_loader
        best = _run_se_tfa_stage(
            model,
            stage_train,
            stage_val,
            config,
            stage_index,
            config.train.stages[stage],
            history,
            output_dir,
            class_names,
            channel_names,
            resume_data if stage_index == resume_stage else None,
        )
        resume_data = None

    return model, {
        "best_val_loss": best["best_loss"],
        "accuracy_at_best_val_loss": best["best_acc"],
        "history": history,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    class_names: List[str],
    channel_names: Optional[List[str]] = None,
    resume: Optional[str] = None,
    next_stage: bool = False,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Dispatch an experiment to its configured training protocol."""
    if next_stage and (config.train.protocol != "se_tfa_paper" or resume is None):
        raise ValueError("--next-stage requires SE-TFA staged training and --resume.")
    if config.train.protocol == "standard":
        return _train_standard(
            model,
            train_loader,
            val_loader,
            config,
            class_names,
            channel_names,
            resume,
        )
    if config.train.protocol == "se_tfa_paper":
        return _train_se_tfa_paper(
            model,
            train_loader,
            val_loader,
            config,
            class_names,
            channel_names,
            resume,
            next_stage=next_stage,
        )
    raise ValueError(f"Unknown training protocol: {config.train.protocol}")
