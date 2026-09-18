# Deep Learning-Based Automatic Modulation Recognition

PyTorch training and evaluation framework for automatic modulation recognition
(AMR) over severe fading channels. The repository currently provides two
reproduced baseline models:

- O'Shea ResNet
- MCLDNN

MATLAB utilities are included for dataset generation and publication-ready
result plotting. Generated datasets, checkpoints, and experiment outputs are
not tracked by Git.

## Repository Structure

```text
src/
|-- amc/
|   |-- data/                 # MATLAB and RadioML data loaders
|   |-- models/               # O'Shea ResNet, MCLDNN, and model factory
|   |-- training/             # Training loop and checkpoint utilities
|   |-- train.py              # Training entry point
|   |-- evaluate.py           # Evaluation and raw-result export
|   `-- benchmark.py          # Accuracy, Params, MACs, and latency summary
|-- configs/                  # Reproducible experiment configurations
|-- dataset_generator_matlab/ # MATLAB dataset generation and visualization
|-- result_plotting_matlab/   # MATLAB result plotting
|-- environment.yml
`-- README.md
```

The dataset generator has its own detailed documentation in
[`dataset_generator_matlab/README.md`](dataset_generator_matlab/README.md).

## Environment

The provided Conda environment uses Python 3.10 and CUDA-enabled PyTorch. From
the `src` directory, create and activate it with:

```bash
conda env create --file environment.yml
conda activate ee6008-amr
```

To update an existing environment:

```bash
conda env update --name ee6008-amr --file environment.yml --prune
```

Verify the interpreter and GPU before training:

```bash
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The CUDA wheel in `environment.yml` must be supported by the installed NVIDIA
driver. MATLAB and MATLAB toolboxes are installed separately.

## Dataset Interface

The baseline configs expect the generated dataset at:

```text
<project-root>/datasets/matlab_amr_dataset.mat
```

The MATLAB v7.3 file is read lazily with `h5py`. Its required variables are:

| Variable | Meaning |
| --- | --- |
| `X` | I/Q samples with shape `[N, 2, L]` |
| `y` | Zero-based modulation labels |
| `snr` | SNR values in dB |
| `channel` | Zero-based channel labels |
| `split` | `0=train`, `1=validation`, `2=test` |
| `classNames` | Modulation names |
| `channelNames` | Channel names |

The current configs use input length 512, 15 modulation classes, 11 channel
conditions, SNR values from -20 dB to 18 dB in 2 dB steps, and a 6:2:2 split.
See the [dataset generator documentation](dataset_generator_matlab/README.md)
for generation, channel definitions, modulation definitions, and dataset
visualization.

## Baseline Models

### O'Shea ResNet

`amc/models/oshea_resnet.py` is a PyTorch reproduction of the residual network
described in:

> T. J. O'Shea, T. Roy, and T. C. Clancy, "Over-the-Air Deep Learning Based
> Radio Signal Classification," *IEEE Journal of Selected Topics in Signal
> Processing*, vol. 12, no. 1, pp. 168-179, Feb. 2018.
> [DOI: 10.1109/JSTSP.2018.2797022](https://doi.org/10.1109/JSTSP.2018.2797022) |
> [arXiv: 1712.04578](https://arxiv.org/abs/1712.04578)

### MCLDNN

`amc/models/mcldnn.py` is a PyTorch reproduction of the multi-channel
convolutional long short-term memory network introduced in:

> J. Xu, C. Luo, G. Parr, and Y. Luo, "A Spatiotemporal Multi-Channel Learning
> Framework for Automatic Modulation Recognition," *IEEE Wireless
> Communications Letters*, vol. 9, no. 10, pp. 1629-1632, Oct. 2020.
> [DOI: 10.1109/LWC.2020.2999453](https://doi.org/10.1109/LWC.2020.2999453) |
> [Official implementation](https://github.com/wzjialang/MCLDNN)

The official MCLDNN implementation uses TensorFlow/Keras; this repository
reimplements the architecture in PyTorch for a unified comparison framework.

Both models use the same data split and training protocol for a controlled
comparison. Following the protocol in arXiv:2606.09085v1, the committed configs
use per-sample L2 normalization, cross-entropy, AdamW with learning rate 0.001
and weight decay 0.01, and `ReduceLROnPlateau` with patience 5 and factor 0.5.
Training uses mixed precision, batch size 512 for the `2 x 512` project signals,
a maximum of 100 epochs, and early stopping with patience 15.

## Training

All commands below use O'Shea ResNet as the primary example. Run commands from
`src`:

```bash
python -m amc.train --config configs/resnet.json
```

Relative data and output paths are resolved from the `src` directory,
independent of the shell's current working directory. The default experiment
output is:

```text
runs/oshea_resnet/
```

Each run stores `best.pt`, `last.pt`, `config.json`, and `history.json`.
`best.pt` is selected by validation loss. Resume an interrupted run from the
complete training state in `last.pt`:

```bash
python -m amc.train --config configs/resnet.json --resume runs/oshea_resnet/last.pt
```

The configs use `device: auto`; CUDA is selected automatically when available.
To train MCLDNN instead, use `configs/mcldnn.json`; its default output directory
is `runs/mcldnn`, and the training and resume commands otherwise remain the
same.

## Evaluation

Evaluate the ResNet test split:

```bash
python -m amc.evaluate \
  --config configs/resnet.json \
  --checkpoint runs/oshea_resnet/best.pt \
  --output-dir runs/eval_oshea_resnet
```

On PowerShell, place each command on one line or replace `\` with the
PowerShell continuation character.

For MCLDNN, replace the config with `configs/mcldnn.json`, the checkpoint with
`runs/mcldnn/best.pt`, and the output directory with `runs/eval_mcldnn`.

Evaluation exports numeric data only:

| File | Contents |
| --- | --- |
| `metrics.json` | Accuracy, macro F1, kappa, and grouped metrics |
| `snr_accuracy.csv` | Overall accuracy at each SNR |
| `modulation_snr_accuracy.csv` | Per-modulation accuracy at each SNR |
| `channel_accuracy.csv` | Accuracy for each channel |
| `channel_snr_accuracy.csv` | Accuracy for each channel and SNR |
| `confusion_matrix.csv` | Raw confusion-matrix sample counts |
| `predictions.mat` | Labels, predictions, SNRs, and channels |
| `tsne_features_0dB.mat` | Balanced features used by MATLAB t-SNE |

Use `--split val` to evaluate the validation split,
`--feature-snr <value>` to change the exported feature SNR, or
`--skip-feature-export` to skip feature collection.

## Model Benchmark

Keep complexity and hardware benchmarking separate from dataset evaluation.
After `evaluate.py` has produced `metrics.json`, summarize the ResNet results:

```bash
python -m amc.benchmark --config configs/resnet.json --checkpoint runs/oshea_resnet/best.pt --metrics runs/eval_oshea_resnet/metrics.json --output runs/eval_oshea_resnet/benchmark.json
```

The command writes `benchmark.json` with full metadata and `benchmark.csv`
with the paper-table columns `ave_acc_pct`, `max_acc_pct`, `params_k`,
`macs_m`, and `latency_ms`. `Ave. Acc.` is the sample-weighted test accuracy
from `metrics.json`; `Max. Acc.` is the highest per-SNR test accuracy. Latency
defaults to batch size 1, FP32, 50 warm-up iterations, and 200 measured
iterations on the configured device. Because latency is hardware-dependent,
the JSON also records the device, precision, batch size, and repetition count.
MACs are always reported for one input sample; one MAC denotes one
multiply-accumulate operation (often approximated as two FLOPs).

Use `--amp` for CUDA mixed-precision latency or change `--batch-size`,
`--warmup`, and `--repetitions` explicitly when reporting another protocol.

## MATLAB Result Plotting

Evaluation deliberately does not create figures. MATLAB reads the exported
CSV and MAT files and writes plots under `result_plotting_matlab/figures`.
See [`result_plotting_matlab/README.md`](result_plotting_matlab/README.md) for
the complete plotting workflow.
