"""Group-member checkpoint compatibility and unified training integration."""

import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from amc.benchmark import count_macs
from amc.config import load_config
from amc.evaluate import final_linear
from amc.models import build_model
from amc.training.engine import (
    _make_optimizer, _make_scheduler, _snr_weighted_loss, _train_standard, evaluate,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "amcnet": ("amc/models/amcnet.py", "AMCNet", None),
    "mcformer": ("amc/models/mcformer.py", "MCformer", None),
    "mccfn": ("mccfn/model/mccfn.py", "MCCFN", "mccfn/output/best.pt"),
    "tldnn": ("tldnn/model/tldnn.py", "TLDNN", "tldnn/output/best.pt"),
    "dhn": ("dhn/dhn/dhn/dhn.py", "DHN", "dhn/dhn/output/best.pt"),
}


class MemberBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def config(self, name):
        return load_config(ROOT / "configs/baselines" / f"{name}.json")

    def test_new_models_train_and_export_classifier_features(self):
        for name in SOURCES:
            with self.subTest(model=name):
                config = self.config(name)
                model = build_model(config.model)
                samples = [{"x": torch.randn(2, 512), "y": torch.tensor(i),
                            "snr": torch.tensor(0), "channel": torch.tensor(0)}
                           for i in range(2)]
                x = torch.stack([sample["x"] for sample in samples])
                logits = model(x)
                self.assertEqual(logits.shape, (2, 15))
                nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters() if p.requires_grad))
                optimizer = _make_optimizer(model, config)
                optimizer.step()
                metrics = evaluate(model, DataLoader(samples, batch_size=2),
                                   torch.device("cpu"), criterion=nn.CrossEntropyLoss(),
                                   num_classes=15, feature_layer=final_linear(model),
                                   feature_snr=0, export_predictions=True)
                self.assertEqual(metrics["_prediction_export"]["y_pred"].shape, (2,))
                self.assertEqual(metrics["_feature_export"]["features"].shape[0], 2)
                self.assertGreater(count_macs(model, (1, 2, 512)), 0)

    def test_original_member_models_and_checkpoints_have_identical_logits(self):
        if not (ROOT / "group_members").is_dir():
            self.skipTest("Original member submissions are not installed.")
        for name, (source, class_name, checkpoint_path) in SOURCES.items():
            with self.subTest(model=name):
                spec = importlib.util.spec_from_file_location(
                    f"member_reference_{name}", ROOT / "group_members" / source)
                reference_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(reference_module)
                config = self.config(name)
                torch.manual_seed(42)
                reference = getattr(reference_module, class_name)(
                    num_classes=15, input_channels=2, input_length=512,
                    hidden_channels=config.model.hidden_channels,
                    dropout=config.model.dropout, **config.model.extra).eval()
                torch.manual_seed(42)
                model = build_model(config.model).eval()
                if checkpoint_path:
                    checkpoint = torch.load(ROOT / "group_members" / checkpoint_path,
                                            map_location="cpu", weights_only=False)
                    model.load_state_dict(checkpoint["model"], strict=True)
                    reference.load_state_dict(checkpoint["model"], strict=True)
                x = torch.randn(2, 2, 512)
                with torch.no_grad():
                    torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)

    def test_dhn_snr_loss_weights_each_sample_and_preserves_gradient(self):
        logits = torch.tensor([[0.8, -0.3], [0.8, -0.3]], requires_grad=True)
        y = torch.tensor([0, 0])
        snr = torch.tensor([-6.0, 18.0])
        loss = _snr_weighted_loss(logits, y, snr, 0.3, 6.0)
        ce = nn.functional.cross_entropy(logits, y, reduction="none")
        weights = torch.sigmoid(torch.tensor([0.0, 7.2]))
        torch.testing.assert_close(loss, (weights * ce).mean())
        gradient = torch.autograd.grad(loss, logits)[0]
        ce_gradient = torch.autograd.grad(ce.mean(), logits)[0]
        torch.testing.assert_close(gradient, ce_gradient * weights[:, None])
        with self.assertRaises(ValueError):
            _snr_weighted_loss(logits, y, torch.tensor([float("nan"), 0]), 0.3, 6)

    def test_dhn_adam_and_step_decay_match_config(self):
        config = self.config("dhn")
        optimizer = _make_optimizer(nn.Linear(2, 2), config)
        self.assertIs(type(optimizer), torch.optim.Adam)
        scheduler = _make_scheduler(optimizer, config)
        for _ in range(9):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.001)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0009)

    def test_weighted_training_logs_ce_separately_and_resumes(self):
        config = deepcopy(self.config("dhn"))
        config.model.name = "test_tiny"
        config.train.device, config.train.amp = "cpu", False
        config.train.epochs, config.train.log_interval = 1, 0
        model = nn.Sequential(nn.Flatten(), nn.Linear(1024, 15))
        samples = [{"x": torch.randn(2, 512), "y": torch.tensor(i),
                    "snr": torch.tensor(-20 + i * 38), "channel": torch.tensor(0)}
                   for i in range(2)]
        loader = DataLoader(samples, batch_size=2)
        snapshots = {}

        def save(payload, filename):
            snapshots[Path(filename).name] = deepcopy(payload)

        with patch("amc.training.engine.save_config"), \
             patch("amc.training.engine.Path.mkdir"), \
             patch("amc.training.engine.Path.write_text"), \
             patch("amc.training.engine.save_checkpoint", side_effect=save), \
             patch("amc.training.engine.load_checkpoint",
                   side_effect=lambda *a, **kw: deepcopy(snapshots["last.pt"])):
            config.train.output_dir = str(ROOT / "tmp/member_baseline_training_test")
            _train_standard(model, loader, loader, config, [str(i) for i in range(15)])
            checkpoint = snapshots["last.pt"]
            row = checkpoint["history"][0]
            self.assertLess(row["train_objective_loss"], row["train_loss"])
            self.assertEqual(checkpoint["config"]["train"]["optimizer"], "adam")
            with torch.no_grad(): expected_ce = nn.functional.cross_entropy(
                model(torch.stack([s["x"] for s in samples])), torch.tensor([0, 1]))
            self.assertAlmostEqual(row["val_loss"], expected_ce.item(), places=6)
            config.train.epochs = 2
            _train_standard(model, loader, loader, config, [str(i) for i in range(15)],
                            resume="last.pt")
            config.train.optimizer = "adamw"
            with self.assertRaisesRegex(ValueError, "optimizer"):
                _train_standard(model, loader, loader, config, [str(i) for i in range(15)],
                                resume="last.pt")

    def test_attention_macs_include_functional_projections_and_products(self):
        class AttentionOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiheadAttention(16, 4, batch_first=True)

            def forward(self, x):
                return self.attention(x, x, x, need_weights=False)[0]

        # Five 16-dimensional tokens: four projections plus QK and AV products.
        expected = 4 * 5 * 16 * 16 + 2 * 5 * 5 * 16
        original = torch.backends.mha.get_fastpath_enabled()
        self.assertEqual(count_macs(AttentionOnly(), (1, 5, 16)), expected)
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(), original)

    def test_feat_is_not_registered(self):
        for name in ("feat", "fea_t"):
            config = deepcopy(self.config("dhn"))
            config.model.name = name
            with self.assertRaisesRegex(ValueError, "Unsupported model"):
                build_model(config.model)


if __name__ == "__main__":
    unittest.main()
