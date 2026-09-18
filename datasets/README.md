# Datasets

Large dataset files are distributed separately and are not committed to this
Git repository.

## MATLAB AMR Dataset

The formal project dataset is stored as a MATLAB v7.3/HDF5 file containing I/Q
signals, modulation labels, SNR values, channel labels, and predefined
train/validation/test splits.

Download the dataset from the shared Google Drive folder:

- [Google Drive - MATLAB AMR Dataset](https://drive.google.com/drive/folders/13ni3EhD0toR7zA0D2i1iJit5bEL0YuD0?usp=sharing)

Download `matlab_amr_dataset.mat` from the folder and keep the file name
unchanged.

After downloading, place the file at:

```text
<project-root>/datasets/matlab_amr_dataset.mat
```

Both baseline configurations already point to this location. The expected
variable interface is documented in [`src/README.md`](../src/README.md), while
the generation procedure and complete dataset specification are documented in
[`src/dataset_generator_matlab/README.md`](../src/dataset_generator_matlab/README.md).

Additional mirrors or checksum information can be added here when available.

## RadioML

The Python data layer can also read supported RadioML files for baseline
experiments. RadioML datasets are not redistributed by this project; obtain
them from their original provider and comply with their respective terms.
