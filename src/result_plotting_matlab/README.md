# MATLAB Result Plotting

This folder contains the MATLAB plotting layer for AMR experiments. Python
evaluation exports raw metrics and features only; these scripts read those
files and save figures under `result_plotting_matlab/figures`.

## 1. Export Evaluation Results

Run evaluation from the `src` directory before plotting. The workflow below
uses O'Shea ResNet as the primary example:

```bash
python -m amc.evaluate --config configs/resnet.json --checkpoint runs/oshea_resnet/best.pt --output-dir runs/eval_oshea_resnet
```

The evaluation directory contains:

```text
metrics.json
snr_accuracy.csv
modulation_snr_accuracy.csv
channel_accuracy.csv
channel_snr_accuracy.csv
confusion_matrix.csv
predictions.mat
tsne_features_0dB.mat
```

The last file is omitted when evaluation uses `--skip-feature-export`.

## 2. Plotting Entry Points

The `run_*.m` files are ready-to-run entry points. Their input paths and output
names are defined near the top of each script and can be changed for another
model or experiment.

| Entry point | Default input | Output |
| --- | --- | --- |
| `run_plot_snr.m` | ResNet `snr_accuracy.csv` | `snr_accuracy_paper.png` |
| `run_plot_snr_compare.m` | ResNet and MCLDNN `snr_accuracy.csv` | `snr_accuracy_compare.png` |
| `run_plot_modulation_snr.m` | ResNet `modulation_snr_accuracy.csv` | `modulation_snr_accuracy.png` |
| `run_plot_confusion_matrix.m` | ResNet `confusion_matrix.csv` | `confusion_matrix.png` |
| `run_plot_tsne.m` | ResNet `tsne_features_0dB.mat` | `tsne_oshea_resnet_0dB.png` |

MCLDNN uses the same result format. To plot it, replace `eval_oshea_resnet`
with `eval_mcldnn` in the relevant entry script and update `outputPath` to a
model-specific filename. `run_plot_snr_compare.m` is the provided two-model
comparison example.

## 3. Generate Figures

Open MATLAB and run:

```matlab
cd <project-root>/src/result_plotting_matlab
run_plot_snr
run_plot_snr_compare
run_plot_modulation_snr
run_plot_confusion_matrix
run_plot_tsne
```

The scripts create the `figures` directory automatically. Default outputs are:

```text
figures/snr_accuracy_paper.png
figures/snr_accuracy_compare.png
figures/modulation_snr_accuracy.png
figures/confusion_matrix.png
figures/tsne_oshea_resnet_0dB.png
figures/tsne_embedding_0dB.mat
```

## Figure Definitions

- `plot_snr_accuracy_paper.m` draws overall accuracy versus SNR, with an inset
  for the high-SNR region. The configured axis uses -20 dB to 18 dB in 2 dB
  steps.
- `plot_modulation_snr_accuracy.m` draws one filled-marker curve for every
  modulation class over the same SNR range.
- `plot_confusion_matrix_results.m` displays raw sample counts, not normalized
  values. Every cell is annotated, including zeros.
- `plot_tsne_features.m` reduces exported penultimate-layer features with
  PCA followed by t-SNE. All samples use circular markers, and color identifies
  the modulation class.
- `amr_class_colors.m` provides a shared muted color palette for modulation
  classes.

Evaluation samples t-SNE features evenly across modulation classes and channel
conditions. By default it exports 300 samples per class at 0 dB. To use another
SNR, re-run evaluation, for example:

```bash
python -m amc.evaluate --config configs/resnet.json --checkpoint runs/oshea_resnet/best.pt --output-dir runs/eval_oshea_resnet --feature-snr 12
```

Then update `featureFile` and `outputPath` in `run_plot_tsne.m`.

MATLAB Statistics and Machine Learning Toolbox is required for `tsne`.
