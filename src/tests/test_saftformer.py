"""Full SAFTFormer geometry, independent numerics and training integration."""

import unittest
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from amc.benchmark import count_macs
from amc.config import load_config
from amc.evaluate import final_linear
from amc.models import build_model
from amc.models.ours_saftformer.vit_splus_1d import (
    InputConditionedSymbolAggregation,
    RotaryAttention, SymmetricTimeFrequencyGLU, TimeRoPE,
)
from amc.training.engine import evaluate


ROOT = Path(__file__).resolve().parents[1]


class SAFTFormerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = load_config(ROOT / "configs/ours_saftformer/full.json")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_best_geometry_cost_and_checkpoint_roundtrip(self):
        model = build_model(self.config.model).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 246734)
        self.assertEqual(count_macs(model, (1, 2, 512)), 18258016)
        self.assertEqual(model.num_patches, 59)
        self.assertEqual(model.prefix_tokens, 5)
        self.assertIsInstance(model.patch_embed, InputConditionedSymbolAggregation)
        self.assertTrue(all(isinstance(b.ffn, SymmetricTimeFrequencyGLU)
                            for b in model.blocks))
        self.assertEqual([b.ffn.hidden_dim for b in model.blocks], [192] * 6)
        for block in model.blocks:
            self.assertEqual(block.ffn.temporal.kernel_sizes, (1, 3, 5, 7))
            self.assertEqual(block.ffn.temporal.split_sizes, (48, 48, 48, 48))
            for conv in block.ffn.temporal.convs:
                self.assertEqual(conv.in_channels, conv.groups)
                self.assertIsNone(conv.bias)
        self.assertEqual([b.drop_path.probability for b in model.blocks], [0.1] * 6)
        shapes = []
        handles = [b.register_forward_hook(lambda _m, _args, y: shapes.append(tuple(y.shape)))
                   for b in model.blocks]
        x = torch.randn(2, 2, 512)
        with torch.no_grad(): expected = model(x)
        for handle in handles: handle.remove()
        self.assertEqual(shapes, [(2, 64, 48)] * 6)
        restored = build_model(self.config.model).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad(): torch.testing.assert_close(restored(x), expected, atol=0, rtol=0)

    def test_frontend_matches_independent_iq_windows_and_position_projections(self):
        embedding = build_model(self.config.model).patch_embed.double()
        with torch.no_grad(): embedding.symbol_score[-1].weight.normal_(std=0.1)
        x = torch.randn(2, 2, 512, dtype=torch.float64, requires_grad=True)
        iq = x.unfold(-1, 8, 8).permute(0, 2, 1, 3).reshape(2, 64, 16)
        symbols = F.silu(F.linear(iq, embedding.symbol_encoder.weight.flatten(1)))
        windows = torch.stack([symbols[:, start:start + 6] for start in range(59)], dim=1)
        relative = windows - windows.mean(2, keepdim=True)
        scores = embedding.symbol_score(torch.cat((windows, relative), -1))
        weighted = windows * (1 + 0.5 * scores.tanh())
        expected = sum(F.linear(weighted[:, :, position],
                                embedding.frame_projection.weight[:, :, position])
                       for position in range(6))
        actual = embedding(x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        actual_grad = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-9)
        self.assertGreater(actual_grad[:, :, -8:].abs().sum().item(), 0)

    def test_attention_matches_explicit_noncausal_matrix_products(self):
        attention = RotaryAttention(16, 2, 3).double().eval()
        rope = TimeRoPE(8).double().eval()(4)
        x = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
        with torch.no_grad(): attention.qkv.bias[16:32].fill_(12)
        bias = attention.qkv.bias.clone()
        bias[16:32] = 0
        qkv = F.linear(x, attention.qkv.weight, bias).reshape(2, 7, 3, 2, 8)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = TimeRoPE.apply_rotary(q, rope, 3)
        k = TimeRoPE.apply_rotary(k, rope, 3)
        weights = (q @ k.transpose(-1, -2) / 8 ** 0.5).softmax(-1)
        expected = attention.proj((weights @ v).transpose(1, 2).reshape(2, 7, 16))
        actual = attention(x, rope)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        actual_grad = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-10)

    def test_rope_preserves_prefix_and_patch_norms(self):
        rope = TimeRoPE(8).double().eval()(59)
        x = torch.randn(2, 6, 64, 8, dtype=torch.float64)
        rotated = TimeRoPE.apply_rotary(x, rope, 5)
        torch.testing.assert_close(rotated[:, :, :5], x[:, :, :5], atol=0, rtol=0)
        torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))

    def test_spectrum_is_finite_and_not_added_to_prefix_tokens(self):
        model = build_model(self.config.model).eval()
        with torch.no_grad():
            spectrum = model.spectral_conditioner.log_power_spectrum(torch.zeros(2, 2, 512))
            frequency = model.spectral_conditioner(torch.zeros(2, 2, 512))
            logits = model(torch.zeros(2, 2, 512))
        self.assertEqual(spectrum.shape, (2, 59, 48))
        self.assertEqual(frequency.shape, (2, 59, 192))
        self.assertTrue(torch.isfinite(spectrum).all())
        self.assertTrue(torch.isfinite(frequency).all())
        self.assertTrue(torch.isfinite(logits).all())
        # Activate the initially neutral condition so this checks separation
        # rather than merely comparing two zero-scaled branches.
        ffn = model.blocks[-1].ffn
        with torch.no_grad(): ffn.frequency_scale.fill_(1)
        x = torch.randn(2, 64, 48)
        with torch.no_grad():
            unconditioned = ffn(x, torch.zeros_like(frequency))
            conditioned = ffn(x, torch.randn_like(frequency))
        torch.testing.assert_close(conditioned[:, :5], unconditioned[:, :5], atol=0, rtol=0)
        self.assertFalse(torch.allclose(conditioned[:, 5:], unconditioned[:, 5:]))

    def test_classification_trains_dense_blocks_and_frontend(self):
        model = build_model(self.config.model).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        watched = [model.patch_embed.symbol_encoder.weight,
                   model.patch_embed.symbol_score[-1].weight,
                   *(block.ffn.a.weight for block in model.blocks)]
        before = [parameter.detach().clone() for parameter in watched]
        x = torch.randn(4, 2, 512)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), torch.arange(4))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            self.assertTrue(all(parameter.grad.abs().sum().item() > 0 for parameter in watched))
            optimizer.step()
        self.assertTrue(all(not torch.equal(old, parameter)
                            for old, parameter in zip(before, watched)))

    def test_evaluation_exports_cls_features_and_predictions(self):
        model = build_model(self.config.model)
        x = torch.randn(2, 2, 512)
        batch = dict(x=x, y=torch.tensor([0, 1]), snr=torch.zeros(2, dtype=torch.long),
                     channel=torch.zeros(2, dtype=torch.long))
        metrics = evaluate(model, [batch], torch.device("cpu"), nn.CrossEntropyLoss(),
                           num_classes=15, class_names=self.config.data.class_names,
                           channel_names=self.config.data.channel_names,
                           feature_layer=final_linear(model), feature_snr=0,
                           export_predictions=True)
        self.assertEqual(metrics["_feature_export"]["features"].shape, (2, 48))
        self.assertEqual(metrics["_prediction_export"]["y_pred"].shape, (2,))
        self.assertNotIn("routing", metrics)

    def test_temporal_kernel_options_reach_convolutions_and_reject_invalid_scales(self):
        settings = deepcopy(self.config.model)
        settings.extra["temporal_kernel_sizes"] = [3]
        model = build_model(settings)
        for block in model.blocks:
            temporal = block.ffn.temporal
            self.assertEqual(temporal.split_sizes, (192,))
            self.assertEqual(len(temporal.convs), 1)
            self.assertEqual(temporal.convs[0].kernel_size, (3,))
            self.assertEqual(temporal.convs[0].groups, 192)
        for kernels in ([], [2], [3, 3], [True, 3, 5], "357"):
            with self.subTest(kernels=kernels):
                settings.extra["temporal_kernel_sizes"] = kernels
                with self.assertRaises(ValueError): build_model(settings)

    def test_removed_architectures_cannot_be_enabled(self):
        for changed in ({"ffn_type": "swiglu"}, {"enable_moe": True},
                        {"embedding_type": "patch"}, {"moe_block_indices": [1, 3, 5]},
                        {"moe_num_experts": 8}, {"symbol_kernel_size": 24},
                        {"icsa_gate_mode": "channelwise"}, {"moe_enable_dispersion": True},
                        {"headwise_attn_output_gate": True}, {"readout_type": "class_query"},
                        {"enable_register_aux": True}, {"enable_rism": True}):
            with self.subTest(changed=changed):
                settings = deepcopy(self.config.model)
                settings.extra.update(changed)
                with self.assertRaises(ValueError): build_model(settings)

        for name in ("saftformer_cls_moe", "saftformer_rism"):
            with self.subTest(model_name=name):
                settings = deepcopy(self.config.model)
                settings.name = name
                with self.assertRaises(ValueError): build_model(settings)


if __name__ == "__main__":
    unittest.main()
