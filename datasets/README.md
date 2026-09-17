# Datasets

Large dataset files are distributed separately and are not committed to this
Git repository.

## MATLAB AMR Dataset

The formal project dataset is stored as a MATLAB v7.3/HDF5 file containing I/Q
signals, modulation labels, SNR values, channel labels, and predefined
train/validation/test splits.

**Download link: coming soon.**

After downloading, place the file at:

```text
<project-root>/datasets/matlab_amr_dataset.mat
```

Both baseline configurations already point to this location. The expected
variable interface is documented in [`src/README.md`](../src/README.md), while
the generation procedure and complete dataset specification are documented in
[`src/dataset_generator_matlab/README.md`](../src/dataset_generator_matlab/README.md).

The download link, mirror information, and SHA-256 checksum will be added here
after the dataset has been uploaded to cloud storage.

## RadioML

The Python data layer can also read supported RadioML files for baseline
experiments. RadioML datasets are not redistributed by this project; obtain
them from their original provider and comply with their respective terms.
