# Deep Learning-Based Algorithms for Automatic Modulation Recognition over Severe Fading Channels

This EE6008 group project studies deep learning-based automatic modulation
recognition (AMR) under additive noise and severe fading. It provides a MATLAB
dataset generator, a PyTorch training and evaluation framework, two reproduced
baseline models, and MATLAB scripts for publication-style result plotting.

## Repository Structure

```text
EE6008/
|-- datasets/                  # Dataset download instructions
|-- ref/                       # Literature index and source links
|-- src/
|   |-- amc/                   # PyTorch data, models, training, and evaluation
|   |-- configs/               # ResNet and MCLDNN experiment configurations
|   |-- dataset_generator_matlab/
|   |-- result_plotting_matlab/
|   |-- environment.yml
|   `-- README.md
|-- .gitignore
`-- README.md
```

Large datasets, checkpoints, experiment runs, generated figures, Python cache
files, and local PDF copies are intentionally excluded from Git.

## Components

### Dataset Generation

The MATLAB generator creates balanced I/Q samples for 15 modulation classes,
11 channel conditions, and multiple SNR levels. It also provides paper-style
time-domain visualization for every channel.

See [dataset generation documentation](src/dataset_generator_matlab/README.md)
for the complete modulation list, channel definitions, MATLAB requirements,
generation settings, stored variables, and visualization workflow.

### PyTorch Framework

The training framework supports lazy loading of MATLAB v7.3 datasets, GPU mixed
precision, checkpoint resume, early stopping, `ReduceLROnPlateau`, and metrics
grouped by modulation, SNR, and channel.

The current baseline models are:

- O'Shea ResNet, reproduced from *Over-the-Air Deep Learning Based Radio Signal
  Classification*.
- MCLDNN, reproduced from *A Spatiotemporal Multi-Channel Learning Framework
  for Automatic Modulation Recognition*.

See [training framework documentation](src/README.md) for model sources,
environment details, configuration fields, and complete commands.

### MATLAB Result Plotting

Python evaluation exports numeric CSV, JSON, and MAT results. MATLAB generates
overall SNR curves, per-modulation SNR curves, model comparisons, confusion
matrices, and t-SNE feature visualizations.

See [result plotting documentation](src/result_plotting_matlab/README.md).

## Dataset

The generated dataset is not stored in this Git repository. It can be
downloaded from the
[shared Google Drive folder](https://drive.google.com/drive/folders/13ni3EhD0toR7zA0D2i1iJit5bEL0YuD0?usp=sharing).
See [`datasets/README.md`](datasets/README.md) for the download and placement
instructions. After downloading, the expected layout is:

```text
EE6008/
`-- datasets/
    `-- matlab_amr_dataset.mat
```

The same file can also be generated locally:

```matlab
cd src/dataset_generator_matlab
run_generate_dataset
```

## Quick Start

Create the Conda environment:

```bash
cd src
conda env create --file environment.yml
conda activate ee6008-amr
```

Train the O'Shea ResNet baseline:

```bash
python -m amc.train --config configs/resnet.json
```

Evaluate its best checkpoint:

```bash
python -m amc.evaluate \
  --config configs/resnet.json \
  --checkpoint runs/oshea_resnet/best.pt \
  --output-dir runs/eval_oshea_resnet
```

PowerShell users can place the evaluation command on one line. MCLDNN uses the
same workflow with `configs/mcldnn.json` and the corresponding run directories.

## References

The papers used for model reproduction and project research are listed in
[`ref/README.md`](ref/README.md). The repository links to publisher or arXiv
pages instead of redistributing local PDF copies.

## License

The source code is released under the [MIT License](LICENSE). Third-party
datasets, papers, software, and dependencies remain subject to their respective
licenses and terms.
