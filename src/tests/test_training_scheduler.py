"""Warmup/cosine boundaries, accumulation and exact checkpoint continuation."""

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from amc.config import ExperimentConfig, config_from_dict
from amc.data.datamodule import SignalDataset
from amc.training.engine import _make_scheduler, train


class SchedulerTests(unittest.TestCase):
    def settings(self):
        config = ExperimentConfig()
        config.train.scheduler = "warmup_cosine"
        config.train.epochs = 5
        config.train.warmup_epochs = 1
        config.train.warmup_start_factor = 0.1
        config.train.lr, config.train.min_lr = 1e-3, 1e-5
        return config

    def test_schedule_boundaries_and_legacy_defaults(self):
        config = self.settings()
        optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=config.train.lr)
        scheduler = _make_scheduler(optimizer, config, 4)
        rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(20):
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"])
        self.assertAlmostEqual(rates[0], 1e-4)
        self.assertAlmostEqual(rates[4], 1e-3)
        self.assertAlmostEqual(rates[12], 5.05e-4)
        self.assertAlmostEqual(rates[20], 1e-5)
        self.assertTrue(all(a < b for a, b in zip(rates[:4], rates[1:5])))
        self.assertTrue(all(a > b for a, b in zip(rates[4:20], rates[5:21])))
        legacy = config_from_dict({"train": {"scheduler": "plateau"}})
        self.assertEqual(legacy.train.warmup_epochs, 0)
        self.assertIsInstance(_make_scheduler(optimizer, legacy), torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_accumulation_partial_batch_and_resumed_training_match(self):
        config = self.settings()
        config.train.epochs, config.train.accumulation_steps = 3, 2
        config.train.early_stop_patience, config.train.log_interval = 100, 0
        config.train.device = "cpu"
        config.train.output_dir = str(Path(__file__).resolve().parents[2] / "tmp")
        config.model.num_classes = 3
        data = SignalDataset(np.random.default_rng(42).normal(size=(9, 2, 8)).astype(np.float32),
                             np.arange(9) % 3, snr=np.zeros(9, dtype=np.int64))
        loader = DataLoader(data, batch_size=2)
        torch.manual_seed(42)
        model = nn.Sequential(nn.Flatten(), nn.Dropout(0.1), nn.Linear(16, 3))
        saved = {}

        def capture(checkpoint, path):
            if str(path).endswith("last.pt"):
                saved[checkpoint["epoch"]] = copy.deepcopy(checkpoint)

        metrics = dict(loss=1.0, accuracy=0.5, macro_f1=0.5)
        with patch("amc.training.engine.evaluate", return_value=metrics), \
             patch("amc.training.engine.save_config"), patch.object(Path, "write_text"), \
             patch("amc.training.engine.save_checkpoint", side_effect=capture):
            _, full = train(model, loader, loader, config, ["a", "b", "c"])
        # Five batches with accumulation=2 produce three optimizer updates/epoch.
        self.assertEqual([saved[e]["scheduler"]["last_epoch"] for e in (1, 2, 3)], [3, 6, 9])
        self.assertAlmostEqual(full["history"][0]["lr_start"], 1e-4)
        self.assertAlmostEqual(full["history"][0]["lr"], 7e-4)
        self.assertAlmostEqual(full["history"][1]["lr_start"], 1e-3)
        self.assertAlmostEqual(full["history"][-1]["lr_next"], 1e-5)
        restored = nn.Sequential(nn.Flatten(), nn.Dropout(0.1), nn.Linear(16, 3))
        with patch("amc.training.engine.evaluate", return_value=metrics), \
             patch("amc.training.engine.save_config"), patch.object(Path, "write_text"), \
             patch("amc.training.engine.save_checkpoint"), \
             patch("amc.training.engine.load_checkpoint", return_value=saved[1]):
            _, resumed = train(restored, loader, loader, config, ["a", "b", "c"], resume="unused.pt")
        for a, b in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        for a, b in zip(full["history"], resumed["history"]):
            for key in ("lr", "lr_start", "lr_next", "train_loss"):
                self.assertEqual(a[key], b[key])
        changed = copy.deepcopy(config)
        changed.train.epochs += 1
        with patch("amc.training.engine.load_checkpoint", return_value=saved[1]), \
             self.assertRaisesRegex(ValueError, "schedule horizon"):
            train(restored, loader, loader, changed, ["a", "b", "c"], resume="unused.pt")

    def test_invalid_warmup_controls(self):
        for key, value in (("warmup_epochs", 5), ("warmup_epochs", -1),
                           ("warmup_epochs", True), ("warmup_start_factor", 0),
                           ("warmup_start_factor", float("nan")), ("min_lr", 0.1)):
            config = self.settings()
            setattr(config.train, key, value)
            optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(()))], lr=config.train.lr)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                _make_scheduler(optimizer, config, 4)


if __name__ == "__main__":
    unittest.main()
