"""Independent validation-loss and validation-accuracy checkpoint selection."""

import io
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from amc.config import ExperimentConfig
from amc.training.engine import _train_standard


def tiny_model():
    return nn.Sequential(nn.Flatten(1), nn.Linear(8, 2))


def tiny_loader():
    signals = torch.arange(32, dtype=torch.float32).reshape(4, 2, 4) / 16
    return DataLoader(
        [dict(x=signals[i], y=i % 2, snr=0, channel=0) for i in range(4)],
        batch_size=2, shuffle=False,
    )


def tiny_config(epochs):
    config = ExperimentConfig()
    config.model.name = "checkpoint_test"
    config.model.input_length = config.data.input_length = 4
    config.model.num_classes = 2
    config.train.device, config.train.amp = "cpu", False
    config.train.epochs = epochs
    config.train.scheduler = "plateau"
    config.train.scheduler_patience, config.train.scheduler_factor = 1, 0.5
    config.train.early_stop_patience, config.train.log_interval = 100, 0
    config.train.output_dir = "runs/test_dual_checkpoint_mock_only"
    return config


def validation_metrics(values):
    return [dict(loss=loss, accuracy=accuracy, macro_f1=accuracy)
            for loss, accuracy in values]


def run_training(config, values, model=None, resume_payload=None):
    """Exercise real optimization and selection, with every file write mocked."""
    model = tiny_model() if model is None else model
    loader, saved = tiny_loader(), []

    def capture(payload, path):
        # Each payload includes live tensors/history, so freeze it at save time.
        saved.append((Path(path).name, deepcopy(payload)))

    with redirect_stdout(io.StringIO()), \
         patch("amc.training.engine.evaluate", side_effect=validation_metrics(values)), \
         patch("amc.training.engine.save_checkpoint", side_effect=capture), \
         patch("amc.training.engine.save_config"), \
         patch("amc.training.engine.Path.mkdir"), \
         patch("amc.training.engine.Path.write_text"), \
         patch("amc.training.engine.load_checkpoint", return_value=deepcopy(resume_payload)):
        model, summary = _train_standard(
            model, loader, loader, config, ["a", "b"],
            resume="last.pt" if resume_payload is not None else None,
        )
    return model, summary, saved


def checkpoints(saved, name):
    return [payload for filename, payload in saved if filename == name]


class DualBestCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(19)

    def test_loss_and_accuracy_select_different_weights_and_ties_keep_first(self):
        values = [(0.9, 0.4), (0.7, 0.45), (0.8, 0.6), (0.7, 0.6)]
        model, summary, saved = run_training(tiny_config(4), values)
        self.assertEqual([s["epoch"] for s in checkpoints(saved, "last.pt")], [1, 2, 3, 4])
        self.assertEqual([s["epoch"] for s in checkpoints(saved, "best_loss.pt")], [1, 2])
        self.assertEqual([s["epoch"] for s in checkpoints(saved, "best.pt")], [1, 2])
        self.assertEqual([s["epoch"] for s in checkpoints(saved, "best_acc.pt")], [1, 2, 3])
        loss_best = checkpoints(saved, "best_loss.pt")[-1]
        compatibility_best = checkpoints(saved, "best.pt")[-1]
        accuracy_best = checkpoints(saved, "best_acc.pt")[-1]
        last = checkpoints(saved, "last.pt")[-1]
        self.assertEqual(loss_best["metrics"]["loss"], 0.7)
        self.assertEqual(loss_best["metrics"]["accuracy"], 0.45)
        self.assertEqual(accuracy_best["metrics"]["loss"], 0.8)
        self.assertEqual(accuracy_best["metrics"]["accuracy"], 0.6)
        self.assertEqual(accuracy_best["best_loss"], 0.7)
        self.assertEqual(accuracy_best["best_acc"], 0.6)
        self.assertEqual(accuracy_best["accuracy_at_best_val_loss"], 0.45)
        self.assertEqual(accuracy_best["best_loss_epoch"], 2)
        self.assertEqual(accuracy_best["best_acc_epoch"], 3)
        self.assertEqual(summary["best_val_loss"], 0.7)
        self.assertEqual(summary["best_val_accuracy"], 0.6)
        self.assertEqual(summary["accuracy_at_best_val_loss"], 0.45)
        self.assertEqual((summary["best_loss_epoch"], summary["best_acc_epoch"]), (2, 3))
        for name, value in loss_best["model"].items():
            torch.testing.assert_close(value, compatibility_best["model"][name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, accuracy_best["model"][name])
                            for name, value in loss_best["model"].items()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, last["model"][name], rtol=0, atol=0)
        self.assertEqual(last["bad_epochs"], 2)

    def test_accuracy_improvements_do_not_reset_loss_early_stop_or_plateau(self):
        config = tiny_config(5)
        config.train.scheduler_patience, config.train.early_stop_patience = 0, 2
        _, summary, saved = run_training(config, [(0.5, 0.4), (0.6, 0.5), (0.7, 0.6)])
        self.assertEqual(len(summary["history"]), 3)
        self.assertEqual([s["epoch"] for s in checkpoints(saved, "best_loss.pt")], [1])
        self.assertEqual([s["epoch"] for s in checkpoints(saved, "best_acc.pt")], [1, 2, 3])
        self.assertEqual(checkpoints(saved, "best_acc.pt")[-1]["epoch"], 3)
        last = checkpoints(saved, "last.pt")[-1]
        self.assertEqual(last["bad_epochs"], 2)
        self.assertEqual(last["scheduler"]["best"], 0.5)
        self.assertEqual(last["optimizer"]["param_groups"][0]["lr"], config.train.lr * 0.5)
        self.assertEqual(summary["best_val_loss"], 0.5)
        self.assertEqual(summary["accuracy_at_best_val_loss"], 0.4)
        self.assertEqual(summary["best_val_accuracy"], 0.6)

    def test_resume_preserves_independent_bests_and_exact_optimizer_continuation(self):
        prefix = [(0.9, 0.4), (0.7, 0.45), (0.8, 0.6), (0.7, 0.6)]
        suffix = [(0.9, 0.59), (0.6, 0.58), (0.65, 0.62)]
        initial = tiny_model().state_dict()
        full_model = tiny_model()
        full_model.load_state_dict(initial)
        full_model, full, _ = run_training(tiny_config(7), prefix + suffix, full_model)
        partial_model = tiny_model()
        partial_model.load_state_dict(initial)
        _, _, first_saves = run_training(tiny_config(4), prefix, partial_model)
        previous = checkpoints(first_saves, "last.pt")[-1]
        resumed_model, resumed, saved = run_training(tiny_config(7), suffix, resume_payload=previous)
        self.assertEqual([s["epoch"] for s in checkpoints(saved, "last.pt")], [5, 6, 7])
        self.assertEqual([s["epoch"] for s in checkpoints(saved, "best_loss.pt")], [6])
        self.assertEqual([s["epoch"] for s in checkpoints(saved, "best.pt")], [6])
        self.assertEqual([s["epoch"] for s in checkpoints(saved, "best_acc.pt")], [7])
        fifth = checkpoints(saved, "last.pt")[0]
        self.assertEqual((fifth["best_loss"], fifth["best_acc"]), (0.7, 0.6))
        self.assertEqual((fifth["best_loss_epoch"], fifth["best_acc_epoch"]), (2, 3))
        self.assertEqual(fifth["accuracy_at_best_val_loss"], 0.45)
        sixth = checkpoints(saved, "best_loss.pt")[-1]
        self.assertEqual((sixth["best_loss"], sixth["best_acc"]), (0.6, 0.6))
        self.assertEqual((sixth["best_loss_epoch"], sixth["best_acc_epoch"]), (6, 3))
        self.assertEqual(sixth["accuracy_at_best_val_loss"], 0.58)
        self.assertEqual(resumed["best_val_accuracy"], 0.62)
        for key in ("best_val_loss", "best_val_accuracy", "accuracy_at_best_val_loss",
                    "best_loss_epoch", "best_acc_epoch"):
            self.assertEqual(full[key], resumed[key], key)
        for name, value in full_model.state_dict().items():
            torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)
        for expected, actual in zip(full["history"], resumed["history"]):
            for key in ("epoch", "train_loss", "val_loss", "val_accuracy", "lr"):
                self.assertEqual(expected[key], actual[key], key)

    def test_legacy_resume_recovers_history_max_without_fabricating_old_weights(self):
        prefix = [(0.9, 0.4), (0.7, 0.45), (0.8, 0.6), (0.7, 0.6)]
        _, _, saved = run_training(tiny_config(4), prefix)
        legacy = checkpoints(saved, "last.pt")[-1]
        # Old best_acc represented the accuracy at the minimum-loss epoch.
        legacy["best_acc"] = 0.45
        for key in ("accuracy_at_best_val_loss", "best_loss_epoch", "best_acc_epoch"):
            legacy.pop(key)
        _, summary, resumed_saves = run_training(
            tiny_config(7), [(0.9, 0.59), (0.6, 0.58), (0.65, 0.62)],
            resume_payload=legacy,
        )
        fifth = checkpoints(resumed_saves, "last.pt")[0]
        self.assertEqual(fifth["best_acc"], 0.6)
        self.assertEqual(fifth["best_acc_epoch"], 3)  # First of the tied historical maxima.
        self.assertEqual(fifth["best_loss_epoch"], 2)
        self.assertEqual(fifth["accuracy_at_best_val_loss"], 0.45)
        # Knowing epoch 3's metrics does not recover epoch 3's model weights.
        self.assertEqual([s["epoch"] for s in checkpoints(resumed_saves, "best_acc.pt")], [7])
        self.assertEqual([s["epoch"] for s in checkpoints(resumed_saves, "best_loss.pt")], [6])
        self.assertEqual([s["epoch"] for s in checkpoints(resumed_saves, "best.pt")], [6])
        self.assertEqual(summary["best_val_accuracy"], 0.62)
        self.assertEqual(summary["accuracy_at_best_val_loss"], 0.58)


if __name__ == "__main__":
    unittest.main()
