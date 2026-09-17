"""Metrics for modulation classification."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        cm[int(target), int(pred)] += 1
    return cm


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    total = cm.sum()
    accuracy = float(np.trace(cm) / max(1, total))

    precision = np.diag(cm) / np.maximum(1, cm.sum(axis=0))
    recall = np.diag(cm) / np.maximum(1, cm.sum(axis=1))
    f1 = 2.0 * precision * recall / np.maximum(1e-12, precision + recall)
    macro_f1 = float(np.mean(f1))

    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = float(np.sum(row * col) / max(1, total * total))
    kappa = float((accuracy - pe) / max(1e-12, 1.0 - pe))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "kappa": kappa,
    }


def per_snr_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for value in sorted(np.unique(snr).tolist()):
        mask = snr == value
        result[int(value)] = float(np.mean(y_true[mask] == y_pred[mask]))
    return result


def per_class_snr_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snr: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> Dict[str, Dict[int, float]]:
    """Compute one SNR-accuracy curve for each ground-truth class."""
    result: Dict[str, Dict[int, float]] = {}
    for class_id in sorted(np.unique(y_true).tolist()):
        name = (
            class_names[int(class_id)]
            if class_names is not None and int(class_id) < len(class_names)
            else str(int(class_id))
        )
        curve: Dict[int, float] = {}
        class_mask = y_true == class_id
        for value in sorted(np.unique(snr[class_mask]).tolist()):
            mask = class_mask & (snr == value)
            curve[int(value)] = float(np.mean(y_true[mask] == y_pred[mask]))
        result[name] = curve
    return result


def per_group_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    group_names: Optional[list[str]] = None,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for value in sorted(np.unique(group).tolist()):
        mask = group == value
        key = (
            group_names[int(value)]
            if group_names is not None and int(value) < len(group_names)
            else str(int(value))
        )
        result[key] = float(np.mean(y_true[mask] == y_pred[mask]))
    return result
