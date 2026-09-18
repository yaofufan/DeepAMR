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
experiments. These datasets were originally released by DeepSig and are not
redistributed by this project.

DeepSig classifies the open RadioML datasets as historical research data from
2016-2017 and notes that they contain known errata. The datasets are distributed
under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license.

- [DeepSig official dataset page](https://www.deepsig.ai/datasets/)
- [RadioML official generation code](https://github.com/radioML/dataset)
- [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/)

The official DeepSig download page asks for a name, affiliation, and valid
email address before providing the download. Direct scripted downloads may be
blocked, so use the browser form.

### RadioML2016.10a

On the DeepSig dataset page, select `RADIOML 2016.10A` and then `Dataset
Download`. The official archive is:

```text
RML2016.10a.tar.bz2
```

Extract it into the project dataset directory:

```bash
tar -xjf RML2016.10a.tar.bz2 -C datasets
```

The expected original data file is `RML2016.10a_dict.pkl`. Files named
`RML2016.10a_dict_optimized.pkl` are converted or optimized copies, not the
original DeepSig filename.

### RadioML2016.10b

The historical RadioML release identifies the larger archive as:

```text
RML2016.10b.tar.bz2
```

Its commonly used extracted file is `RML2016.10b.dat`. The current DeepSig
dataset page does not provide a separate 2016.10b download button. Third-party
paper repositories and cloud-storage copies should therefore be described as
mirrors rather than official DeepSig downloads, and their contents should be
verified before use.

### RadioML2018.01A

On the DeepSig dataset page, select `RADIOML 2018.01A` and then `Dataset
Download`. The official archive is:

```text
2018.01.OSC.0001_1024x2M.h5.tar.gz
```

Extract it into the project dataset directory:

```bash
tar -xzf 2018.01.OSC.0001_1024x2M.h5.tar.gz -C datasets
```

The HDF5 dataset is associated with T. J. O'Shea, T. Roy, and T. C. Clancy,
"Over-the-Air Deep Learning Based Radio Signal Classification," *IEEE Journal
of Selected Topics in Signal Processing*, 2018.
[DOI: 10.1109/JSTSP.2018.2797022](https://doi.org/10.1109/JSTSP.2018.2797022)
