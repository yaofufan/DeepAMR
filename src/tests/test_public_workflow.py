"""Public configuration compatibility and trustworthy evaluation records."""

import hashlib
import io
import json
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from amc.config import as_dict, config_from_dict, load_config
from amc.evaluate import main as evaluate_main
from amc.models import build_model


ROOT = Path(__file__).resolve().parents[1]


class PublicWorkflowTests(unittest.TestCase):
    def full_config(self):
        return load_config(ROOT / "configs/ours_saftformer/full.json")

    def test_complete_legacy_metadata_matches_current_full_configuration(self):
        current = self.full_config()
        legacy = deepcopy(as_dict(current))
        legacy["model"]["name"] = "saftformer_legacy"
        legacy["model"]["extra"].update(
            enable_temporal_condition=True, enable_frequency_condition=True,
            enable_moe=False, moe_num_experts=4,
        )
        converted = config_from_dict(legacy)
        self.assertEqual(as_dict(converted)["model"], as_dict(current)["model"])
        self.assertEqual(legacy["model"]["name"], "saftformer_legacy")
        self.assertIn("enable_temporal_condition", legacy["model"]["extra"])

    def test_partial_or_expert_metadata_cannot_load_as_full(self):
        for key in ("enable_temporal_condition", "enable_frequency_condition",
                    "enable_icsa", "enable_moe"):
            legacy = deepcopy(as_dict(self.full_config()))
            legacy["model"]["name"] = "saftformer_legacy"
            legacy["model"]["extra"].update(
                enable_temporal_condition=True, enable_frequency_condition=True,
                enable_icsa=True, enable_moe=False,
            )
            legacy["model"]["extra"][key] = key == "enable_moe"
            with self.subTest(setting=key), self.assertRaises(ValueError):
                build_model(config_from_dict(legacy).model)

    def test_evaluation_record_tracks_used_weights_split_and_sample_count(self):
        config = self.full_config()
        config.train.device = "cpu"
        model = nn.Sequential(nn.Flatten(), nn.Linear(1024, 15))
        loader = DataLoader([
            {"x": torch.zeros(2, 512), "y": torch.tensor(i),
             "snr": torch.tensor(0), "channel": torch.tensor(0)}
            for i in (0, 1)
        ], batch_size=2)
        bundle = SimpleNamespace(
            test=loader, val=loader, class_names=config.data.class_names,
            channel_names=config.data.channel_names,
        )
        checkpoint_bytes = b"evaluation checkpoint fixture"
        for split, filename in (("test", "best.pt"), ("val", "best_acc.pt")):
            writes = {}

            def capture_write(path, text, **kwargs):
                writes[path.name] = json.loads(text)
                return len(text)

            args = SimpleNamespace(
                config=str(ROOT / "configs/ours_saftformer/full.json"),
                checkpoint=f"runs/public_workflow_fixture/{filename}",
                output_dir="runs/public_workflow_fixture/evaluation",
                num_workers=None, skip_feature_export=True, split=split,
                feature_snr=0,
            )
            with self.subTest(split=split), \
                 patch("amc.evaluate.parse_args", return_value=args), \
                 patch("amc.evaluate.load_config", return_value=deepcopy(config)), \
                 patch("amc.evaluate.build_dataloaders", return_value=bundle), \
                 patch("amc.evaluate.build_model", return_value=model), \
                 patch("amc.evaluate.load_checkpoint", return_value={
                     "model": model.state_dict(), "epoch": 7}), \
                 patch("amc.evaluate.write_csv_results"), \
                 patch("amc.evaluate.savemat"), \
                 patch.object(Path, "mkdir"), \
                 patch.object(Path, "open", return_value=io.BytesIO(checkpoint_bytes)), \
                 patch.object(Path, "write_text", autospec=True, side_effect=capture_write), \
                 redirect_stdout(io.StringIO()):
                evaluate_main()
            record = writes["evaluation_provenance.json"]
            self.assertEqual(record["checkpoint_sha256"], hashlib.sha256(checkpoint_bytes).hexdigest())
            self.assertEqual(Path(record["checkpoint_file"]).name, filename)
            self.assertEqual(record["checkpoint_selection"], filename)
            self.assertEqual(record["checkpoint_epoch"], 7)
            self.assertEqual(record["split"], split)
            self.assertEqual(record["evaluated_samples"], 2)
            self.assertEqual(record["test_samples"], 2 if split == "test" else 0)
            self.assertEqual(sum(map(sum, writes["metrics.json"]["confusion_matrix"])), 2)


if __name__ == "__main__":
    unittest.main()
