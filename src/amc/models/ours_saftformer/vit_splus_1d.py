"""SAFTFormer: Symbol-Aware Frequency-Temporal Transformer.

Project full architecture: SHE + scalar ICSA, six dense STF-GLU blocks
with temporal kernels 1/3/5/7, CLS classification.
Input [B,2,512] -> 59 patch + 5 prefix tokens, six layers, width 48, six heads.

DINOv3 backbone reference: https://github.com/facebookresearch/dinov3
STF-GLU combines FEA-T's symmetric DB-GLU with time/frequency conditions;
SiLU and the borrowed mechanisms are not themselves claims of novelty.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .signal_layers import (DropPath, DualBranchGLU, FrameSpectralConditioner,
                            MixedTemporalDWConv, SymbolHierarchicalEmbedding,
                            TimeFrequencyDBGLU, temporal_kernels)


class InputConditionedSymbolAggregation(SymbolHierarchicalEmbedding):
    """Scalar content/deviation gates before the position-specific SHE projection."""

    def __init__(self, frame_length: int, frame_stride: int, dim: int,
                 symbol_length: int = 8, symbol_channels: int = 16,
                 hidden_channels: int = 8, gate_scale: float = 0.5) -> None:
        super().__init__(frame_length, frame_stride, dim, symbol_length, symbol_channels)
        self.gate_scale = gate_scale
        self.symbol_score = nn.Sequential(
            nn.Linear(2 * symbol_channels, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, 1))
        self.reset_conditioning()

    def reset_conditioning(self) -> None:
        # Called again after the enclosing model's generic Linear initialization.
        nn.init.zeros_(self.symbol_score[-1].weight)
        nn.init.zeros_(self.symbol_score[-1].bias)

    def aggregation_weights(self, windows: torch.Tensor) -> torch.Tensor:
        """Return [B,frames,symbols,1] gates for [B,frames,symbols,C]."""
        relative = windows - windows.mean(dim=2, keepdim=True)
        scores = self.symbol_score(torch.cat((windows, relative), dim=-1))
        return 1 + self.gate_scale * scores.tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[-1] < self.frame_length:
            raise ValueError(f"ICSA expects [B,2,L] with L >= {self.frame_length}.")
        symbols = self.activation(self.symbol_encoder(x))
        span, step = self.frame_projection.kernel_size[0], self.frame_projection.stride[0]
        windows = symbols.unfold(-1, span, step).permute(0, 2, 3, 1)
        gated = windows * self.aggregation_weights(windows)
        batch, frames, _, channels = gated.shape
        # Fold frames into batch so the original Conv1d remains visible to MAC
        # profiling and retains channel-major / symbol-position weight ordering.
        projected = self.frame_projection(
            gated.permute(0, 1, 3, 2).reshape(batch * frames, channels, span))
        return projected.squeeze(-1).reshape(batch, frames, -1)



class KeyMaskedLinear(nn.Linear):
    """Learn Q/V biases and mask the key bias to zero."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim, 3 * dim)
        mask = torch.ones(3 * dim)
        mask[dim:2 * dim] = 0
        self.register_buffer("bias_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias * self.bias_mask)



class TimeRoPE(nn.Module):
    """Weight-free rotations of temporal patch tokens; prefix tokens stay intact."""

    def __init__(self, head_dim: int, base: float = 100.0,
                 rescale_coords: float = 2.0) -> None:
        super().__init__()
        if head_dim < 2 or head_dim % 2:
            raise ValueError("RoPE needs a positive even head dimension.")
        if not math.isfinite(base) or base <= 0:
            raise ValueError("rope_base must be finite and positive.")
        if not math.isfinite(rescale_coords) or rescale_coords < 1:
            raise ValueError("rope_rescale_coords must be finite and >= 1.")
        self.rescale_coords = rescale_coords
        # One temporal axis uses all D/2 rotation pairs, unlike two image axes.
        periods = base ** (torch.arange(head_dim // 2, dtype=torch.float32)
                           / (head_dim // 2))
        self.register_buffer("periods", periods)

    def forward(self, num_patches: int) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.float64 if self.periods.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=self.periods.device.type, enabled=False):
            centres = (torch.arange(num_patches, device=self.periods.device,
                                    dtype=dtype) + 0.5) / num_patches
            coords = centres * 2 - 1
            if self.training and self.rescale_coords > 1:
                extent = math.log(self.rescale_coords)
                scale = torch.empty((), device=coords.device, dtype=dtype).uniform_(-extent, extent).exp()
                coords = coords * scale
            phase = 2 * math.pi * coords[:, None] / self.periods.to(dtype)[None, :]
            phase = torch.cat((phase, phase), dim=-1)
            return phase.sin(), phase.cos()

    @staticmethod
    def apply_rotary(x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor],
              prefix_tokens: int) -> torch.Tensor:
        sin, cos = rope
        with torch.autocast(device_type=x.device.type, enabled=False):
            patches = x[:, :, prefix_tokens:].to(sin.dtype)
            first, second = patches.chunk(2, dim=-1)
            rotated = torch.cat((-second, first), dim=-1)
            patches = (patches * cos + rotated * sin).to(x.dtype)
            return torch.cat((x[:, :, :prefix_tokens], patches), dim=-2)


class RotaryAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, prefix_tokens: int,
                 dropout: float = 0.0, attention_dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.prefix_tokens = prefix_tokens
        self.attention_dropout = attention_dropout
        self.qkv = KeyMaskedLinear(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = TimeRoPE.apply_rotary(q, rope, self.prefix_tokens)
        k = TimeRoPE.apply_rotary(k, rope, self.prefix_tokens)
        result = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        result = result.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(result))


class SymmetricTimeFrequencyGLU(TimeFrequencyDBGLU):
    """STF-GLU: Symmetric Time-Frequency Gated Linear Unit.

Apply local time/frequency conditions only to the temporal patch suffix.

Prefix tokens use the same DB-GLU projections without local conditions. They
receive signal information through global attention, not a fictitious time grid.
"""

    def __init__(self, dim: int, hidden_dim: int, prefix_tokens: int,
                 dropout: float, kernel_sizes: Sequence[int]) -> None:
        super().__init__(dim, hidden_dim, dropout, kernel_sizes)
        self.hidden_dim = hidden_dim
        self.prefix_tokens = prefix_tokens

    def forward(self, x: torch.Tensor,
                frequency_context: torch.Tensor | None = None) -> torch.Tensor:
        prefix = DualBranchGLU.forward(self, x[:, :self.prefix_tokens])
        patches = super().forward(x[:, self.prefix_tokens:], frequency_context)
        return torch.cat((prefix, patches), dim=1)


class RotaryBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, prefix_tokens: int,
                 hidden_dim: int, norm_eps: float, layerscale_init: float,
                 dropout: float, attention_dropout: float, drop_path: float,
                 temporal_kernel_sizes: Sequence[int]) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = RotaryAttention(dim, num_heads, prefix_tokens, dropout, attention_dropout)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn = SymmetricTimeFrequencyGLU(dim, hidden_dim, prefix_tokens, dropout, temporal_kernel_sizes)
        self.attn_scale = nn.Parameter(torch.full((dim,), layerscale_init))
        self.ffn_scale = nn.Parameter(torch.full((dim,), layerscale_init))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor],
                frequency_context: torch.Tensor | None) -> torch.Tensor:
        x = x + self.drop_path(self.attn_scale * self.attn(self.norm1(x), rope))
        output = self.ffn(self.norm2(x), frequency_context)
        return x + self.drop_path(self.ffn_scale * output)



BEST_EXTRA = {
    "patch_size": 48, "patch_stride": 8, "depth": 6, "num_heads": 6,
    "ffn_ratio": 6, "num_register_tokens": 4, "norm_eps": 1e-5,
    "layerscale_init": 1e-5, "rope_base": 100, "rope_rescale_coords": 2,
    "attention_dropout": 0, "drop_path_rate": 0.1, "drop_path_schedule": "uniform",
    "ffn_type": "stf_glu", "ffn_activation": "silu",
    "embedding_type": "symbol_hierarchical", "symbol_length": 8,
    "symbol_channels": 16, "embedding_activation": "silu",
    "enable_icsa": True, "icsa_hidden_channels": 8, "icsa_gate_scale": 0.5,
    "temporal_kernel_sizes": [1, 3, 5, 7], "frequency_kernel_size": 3,
    "frequency_channels": 8, "frequency_window": "hann", "frequency_eps": 1e-6,
}


class ViTSPlus1D(nn.Module):
    """SAFTFormer full model; retain the saved class and parameter names."""

    FULL_NAME = "Symbol-Aware Frequency-Temporal Transformer for Automatic Modulation Recognition"

    def __init__(self, num_classes: int, input_channels: int = 2,
                 input_length: int = 512, hidden_channels: int = 48,
                 dropout: float = 0.0,
                 temporal_kernel_sizes: Sequence[int] = (1, 3, 5, 7), **settings) -> None:
        super().__init__()
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer.")
        kernel_sizes = temporal_kernels(temporal_kernel_sizes)
        # Disabled experiment fields in older dense checkpoints are metadata
        # only. Active or standalone expert settings must never change the model.
        if settings.get("enable_moe") is False:
            settings = {key: value for key, value in settings.items()
                        if key != "enable_moe" and not key.startswith("moe_")}
        expected = dict(BEST_EXTRA, input_channels=2, input_length=512,
                        hidden_channels=48, dropout=0)
        supplied = dict(settings, input_channels=input_channels, input_length=input_length,
                        hidden_channels=hidden_channels, dropout=dropout)
        for key, value in supplied.items():
            if key not in expected:
                raise ValueError(f"Unsupported SAFTFormer setting: {key}; only the best configuration is retained.")
            target = expected[key]
            if isinstance(target, list):
                matches = isinstance(value, (list, tuple)) and list(value) == target
            else:
                matches = value == target and (isinstance(value, bool) == isinstance(target, bool))
            if not matches:
                raise ValueError(f"SAFTFormer requires {key}={target!r}; received {value!r}.")
        self.input_channels, self.input_length = 2, 512
        self.num_register_tokens, self.prefix_tokens = 4, 5
        self.num_patches, self.num_tokens = 59, 64
        self.ffn_hidden_dim = 192
        self.temporal_kernel_sizes = kernel_sizes
        self.model_name = "SAFTFormer"
        self.model_full_name = self.FULL_NAME
        self.patch_embed = InputConditionedSymbolAggregation(48, 8, 48)
        self.cls_token = nn.Parameter(torch.empty(1, 1, 48))
        self.register_tokens = nn.Parameter(torch.empty(1, 4, 48))
        self.rope = TimeRoPE(8, 100, 2)
        # Initialize common weights using the four-scale reference recipe.
        # Replacing only zero-initialized temporal kernels below keeps all other
        # tensors and the global RNG identical across paired kernel studies.
        reference_kernels = (1, 3, 5, 7)
        self.blocks = nn.ModuleList([
            RotaryBlock(48, 6, 5, 192, 1e-5, 1e-5, 0.0, 0.0, 0.1, reference_kernels)
            for _ in range(6)])
        self.spectral_conditioner = FrameSpectralConditioner(48, 8, 192, 8, 3, 1e-6)
        self.norm = nn.LayerNorm(48, eps=1e-5)
        self.classifier = nn.Linear(48, num_classes)
        self.apply(self._initialize)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.register_tokens, std=0.02)
        if kernel_sizes != reference_kernels:
            with torch.random.fork_rng(devices=[]):
                for block in self.blocks:
                    block.ffn.temporal = MixedTemporalDWConv(192, kernel_sizes)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, InputConditionedSymbolAggregation):
            module.reset_conditioning()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (2, 512):
            raise ValueError(f"Expected IQ [B,2,512], got {tuple(x.shape)}.")
        patches = self.patch_embed(x)
        frequency_context = (self.spectral_conditioner(x)
                             if self.spectral_conditioner is not None else None)
        batch = patches.shape[0]
        tokens = torch.cat((self.cls_token.expand(batch, -1, -1).to(patches.dtype),
                            self.register_tokens.expand(batch, -1, -1).to(patches.dtype),
                            patches), dim=1)
        rope = self.rope(self.num_patches)
        for block in self.blocks:
            tokens = block(tokens, rope, frequency_context)
        return self.norm(tokens)

    @property
    def feature_layer(self) -> nn.Module:
        return self.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x)[:, 0])
