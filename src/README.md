# AMR training and evaluation framework

The `amc` Python package trains and evaluates the selected SAFTFormer model
and nine comparison models using the same data interface. Run the commands
below from `src/`; configured paths are resolved against this directory.

## Setup and data

```bash
conda env create --file environment.yml
conda activate ee6008-amr
```

The environment file describes the working Windows/PyTorch CUDA 12.8 setup;
an appropriate NVIDIA driver is needed for GPU execution. MATLAB and required
toolboxes are installed separately. CPU execution is also supported; see
`--device` on the benchmark command.

Place the MATLAB v7.3 file at `../datasets/matlab_amr_dataset.mat`. The lazy
HDF5 loader reads `X`, `y`, `snr`, `channel`, and saved `split` labels; model
inputs are I/Q tensors shaped `[batch, 2, 512]`. `classNames` and
`channelNames` describe the metadata. See the [dataset instructions](../datasets/README.md)
and [generator guide](dataset_generator_matlab/README.md). The file and model
checkpoints are distributed separately from this repository.

## Train

```bash
python -m amc.train --config configs/ours_saftformer/full.json --no-resume
```

The Full configuration uses AdamW, learning rate `1e-3`, batch size 512,
validation-loss plateau scheduling, early stopping and CUDA AMP when available.
Training writes under `runs/saftformer_full/`: `best.pt` is the validation-loss
selection, `best_acc.pt` records the validation-accuracy selection, and
`last.pt` contains resume state. The complete architecture and training settings
are explained in the [SAFTFormer configuration guide](configs/ours_saftformer/README.md).

`--no-resume` starts a fresh run. With the default Full config, omitting the
flag resumes that run's `last.pt` when it exists. To resume a custom config,
provide `--resume path/to/last.pt` explicitly. Only load trusted checkpoints.

## Evaluate

```bash
python -m amc.evaluate --config configs/ours_saftformer/full.json --checkpoint runs/saftformer_full/best.pt --output-dir runs/eval_saftformer_full
```

Evaluation defaults to the test split and outputs `metrics.json`, accuracy CSVs
by SNR/modulation/channel, `confusion_matrix.csv`, `predictions.mat`, and
`evaluation_provenance.json` with the checkpoint hash and evaluation conditions.
Feature MAT files are optional. Use `--split val` for validation, `--num-workers 0` where
multiprocessing is unavailable, or `--skip-feature-export` to skip features.
Paths default to `best.pt` and `eval_<training-directory>` for the selected
config when not supplied.

## Model benchmark

```bash
python -m amc.benchmark --config configs/ours_saftformer/full.json --checkpoint runs/saftformer_full/best.pt --metrics runs/eval_saftformer_full/metrics.json --output runs/eval_saftformer_full/benchmark.json
```

The CLI reports parameters, per-sample MACs and measured forward latency. Its
ordinary defaults are batch size 1, FP32, 50 warm-ups and 200 measured passes;
set `--device`, `--warmup`, `--repetitions` and `--batch-size` for another
protocol. The README comparison instead used 200 warm-ups, 1,000 measured
passes and the median of three round means on the RTX 5060 Laptop GPU.

Latency is hardware-specific, includes model-internal transforms, and excludes
data loading and host transfer.

## Baselines

Nine comparison implementations live in `amc/models/baselines/` with configs
under `configs/baselines/`. For example:

```bash
python -m amc.train --config configs/baselines/resnet.json --no-resume
python -m amc.evaluate --config configs/baselines/resnet.json --checkpoint runs/oshea_resnet/best.pt --output-dir runs/eval_oshea_resnet
```

The [baseline index](configs/baselines/README.md) maps model names to their
configuration files. Comparison results are shown in the
[repository README](../README.md#results).

## Plotting and tests

MATLAB plotting scripts read evaluation outputs and generate figures locally;
see [the plotting guide](result_plotting_matlab/README.md). Checkpoints, full
runs, exported features, cached embeddings, and original group-member submissions
are excluded from Git. Obtain or generate inputs before invoking a plot.

Run the Python test suite without the dataset:

```bash
python -m unittest discover -s tests -v
```

The optional member-checkpoint equivalence test skips when the local
`group_members/` archive is not present.
