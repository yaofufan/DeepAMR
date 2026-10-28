# SAFTFormer full model configuration

[full.json](full.json) configures **SAFTFormer: Symbol-Aware Frequency-Temporal
Transformer for Automatic Modulation Recognition**. Its three sections define
the input data, the complete network and the training procedure.

## Input data

The MATLAB loader reads `X`, `y`, `snr`, `channel` and `split` from
`../datasets/matlab_amr_dataset.mat`. Each network input is `[B, 2, 512]`:
the two channels contain I and Q. The task has 15 modulation classes and 11
channel conditions. Saved split labels define training, validation and test
membership; the configured proportions are 60%, 20% and 20%.

| Setting | Value |
| --- | --- |
| Input normalization | `none` |
| Batch size | 512 |
| Loader workers | 4 |
| Prefetch factor | 2 |
| Persistent workers | Enabled |
| HDF5 block size | 64 |
| Seed | 42 |

## Network design

The model registry name is `vit_splus_1d`; the published model is SAFTFormer.

| Stage | Configuration | Output shape |
| --- | --- | --- |
| Symbol encoder (SHE) | Conv1d `2 -> 16`, kernel and stride 8, no bias, SiLU | `[B, 16, 64]` (channels, signal units) |
| Frame grouping | Six units per frame; one-unit hop (48 samples, stride 8) | `[B, 59, 6, 16]` |
| Input-conditioned aggregation (ICSA) | Concatenate each unit and its deviation from the frame mean; scorer `32 -> 8 -> 1`, SiLU, gate `1 + 0.5*tanh(score)` | `[B, 59, 6, 16]` |
| Frame projection | Position-specific projection of six gated units, `96 -> 48`, no bias | `[B, 59, 48]` |
| Prefix tokens | One learned CLS and four learned register tokens | `[B, 64, 48]` |
| Transformer encoder | Six blocks, width 48, six attention heads (8 dimensions per head) | `[B, 64, 48]` |
| Classification | Final LayerNorm, select CLS, linear `48 -> 15` | `[B, 15]` logits |

Each encoder block has a pre-normalized attention sublayer followed by a
pre-normalized STF-GLU sublayer. LayerNorm uses `norm_eps=1e-5`. Both residual
branches have LayerScale initialized to `1e-5`, and DropPath uses probability
0.1 in every block (`drop_path_schedule="uniform"`). Ordinary dropout and
attention dropout are both zero.

Temporal RoPE rotates the frame-token queries and keys. Its base is 100;
training samples a log-uniform coordinate scale between 0.5 and 2
(`rope_rescale_coords=2`). CLS and register tokens retain their unrotated
queries and keys. Attention is bidirectional.

### Symmetric time-frequency gating

Each STF-GLU has two linear branches, `a: 48 -> 192` and `b: 48 -> 192`,
followed by a `192 -> 48` output projection. The saved `ffn_ratio=6` is a
compatibility field from the inherited gated-FFN configuration. The current
Full implementation fixes each branch at 192 features (four times the backbone
width).

For frame tokens, the gates use:

```text
u = a + T(a) + gamma * F
v = b + T(b) + gamma * F
output = projection(a * SiLU(v) + b * SiLU(u))
```

`T` splits 192 channels into four groups of 48. Bias-free depthwise temporal
convolutions use kernels `[1, 3, 5, 7]`, preserve token length through padding,
and concatenate the groups. The two branches share this temporal operator
within each block.

`F` is the aligned spectral condition. The original complex signal `I+jQ` is
framed with length 48 and stride 8, then multiplied by a Hann window. An
orthonormal FFT and FFT shift retain all 48 complex-baseband frequency bins.
Power is normalized by its mean within each frame and converted to log power
with `frequency_eps=1e-6`. A bias-free `1 -> 8` pointwise convolution,
SiLU, kernel-3 depthwise convolution, SiLU and `8 -> 1` pointwise convolution
process the frequency bins. A `48 -> 192` linear projection produces
`F` with shape `[B, 59, 192]`.

The spectral network runs once per input. Each encoder block has a separate
learned scalar `gamma`. Temporal kernels and these scalars start at zero;
ICSA's final scoring layer starts at zero, giving unit gates initially.
CLS and register tokens use the same symmetric SiLU projections without
local temporal or spectral conditions. Classification reads the final CLS
feature; the final block's patch-only FFN updates have no subsequent path to
that classifier.

## Training procedure

| Setting | Value / behavior |
| --- | --- |
| Optimizer | AdamW |
| Initial learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Maximum epochs | 100 |
| Learning-rate scheduler | ReduceLROnPlateau on validation loss; patience 5, factor 0.5 |
| Minimum learning rate | `1e-5` |
| Warmup | Disabled (`warmup_epochs=0`) |
| Early stopping | Validation-loss patience 15 |
| Gradient clipping | Maximum norm 5 |
| Gradient accumulation | One batch per optimizer step |
| Mixed precision | AMP on CUDA when available |
| Training objective | Cross-entropy |
| Seed | 42 |
| Training protocol | `standard`; no staged schedule (`stages={}`) |
| Output directory | `runs/saftformer_full` |

`warmup_start_factor=0.1` is inactive while warmup is disabled.
`device="auto"` selects the available device. Training logs every 100 steps.
`best.pt` selects the lowest validation loss, `best_acc.pt` records the highest
validation accuracy, and `last.pt` contains the state needed to resume.

## Run the model

From `src/`:

```bash
python -m amc.train --config configs/ours_saftformer/full.json --no-resume
python -m amc.evaluate --config configs/ours_saftformer/full.json --checkpoint runs/saftformer_full/best.pt --output-dir runs/eval_saftformer_full
python -m amc.benchmark --config configs/ours_saftformer/full.json --checkpoint runs/saftformer_full/best.pt --metrics runs/eval_saftformer_full/metrics.json --output runs/eval_saftformer_full/benchmark.json
```

With the default Full configuration, training resumes its own `last.pt` when
present; use `--no-resume` for a new run. The dataset and checkpoints are
obtained separately. See the [repository README](../../../README.md) for the
architecture figure and comparison results, and the
[framework guide](../../README.md) for environment setup and command options.
