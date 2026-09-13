# Deep Learning-Based Algorithms for Automatic Modulation Recognition over Severe Fading Channels

This project studies deep learning for automatic modulation recognition over
severe fading channels. The current public release contains the MATLAB dataset
generation and I/Q visualization code. Model, training, and evaluation code
are not included in this release.

## Dataset Generator

See [the generator documentation](src/datsetgenarator_matlab/README.md) for
MATLAB requirements, modulation and channel definitions, dataset format,
and generation and visualization instructions.

From the repository root, run in MATLAB:

```matlab
cd src/datsetgenarator_matlab
run_generate_dataset
run_visualize_dataset
```

Generation writes `datasets/matlab_amr_dataset.mat` under the repository root.
The default configuration replaces an existing dataset at that location;
set `cfg.overwriteExisting=false` to prohibit replacement.
Generated datasets and visualization outputs are not tracked in Git.
