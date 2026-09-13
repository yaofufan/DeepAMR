# MATLAB AMR Dataset Generator

This folder owns the formal MATLAB dataset used by the EE6008 automatic
modulation recognition project. It contains one fixed configuration, a
disk-backed generator, and a paper-style I/Q visualization pipeline.

## Files

- `amr_dataset_config.m`: the single formal dataset configuration.
- `generate_amr_dataset.m`: modulation, channel, noise, split, and incremental
  `.mat` writing implementation.
- `run_generate_dataset.m`: one-command generation entry point.
- `visualize_amr_dataset.m`: indexed I/Q time-domain visualization.
- `run_visualize_dataset.m`: one-command visualization entry point.

## Requirements

The formal configuration targets MATLAB R2023b and requires 5G Toolbox for
`nrTDLChannel`. Check it before starting the long generation run:

```matlab
ver('5g')
exist('nrTDLChannel', 'class')
```

The second command must return `8`. The generator stops before creating the
dataset when the formal TDL channels are selected but `nrTDLChannel` is not
available. For a small nonstandard pipeline test only, setting
`cfg.require5GToolbox=false` enables the local approximate TDL fallback.

Root-raised-cosine pulse shaping uses `rcosdesign` when it is available. The
generator otherwise falls back to rectangular symbol repetition.

## Formal Dataset Specification

| Setting | Value |
|---|---:|
| Modulation classes | 15 |
| Channel conditions | 11 |
| SNR values | `-20:2:18` dB (20 points) |
| Samples per modulation-channel-SNR cell | 1000 |
| Signal shape | `[2, 512]` (`I`, `Q`) |
| Samples per symbol | 8 |
| Train/validation/test | 6:2:2 inside every cell |
| Total samples | 3,300,000 |
| Output | `../../datasets/matlab_amr_dataset.mat` |
| Format | MATLAB `-v7.3` (HDF5) |

## Modulation Dictionary

The links point to the closest MathWorks references. The generator implements
lightweight complex-baseband versions locally; it does not call every linked
function directly.

| ID (`y`) | Modulation | Family | Generator definition |
|---:|---|---|---|
| 0 | [BPSK](https://www.mathworks.com/help/comm/ref/pskmod.html) | PSK | 2 equally spaced phases, phase offset `pi/2` |
| 1 | [QPSK](https://www.mathworks.com/help/comm/ref/pskmod.html) | PSK | 4 equally spaced phases, phase offset `pi/4` |
| 2 | [8PSK](https://www.mathworks.com/help/comm/ref/pskmod.html) | PSK | 8 equally spaced phases, phase offset `pi/8` |
| 3 | [2FSK](https://www.mathworks.com/help/comm/ref/fskmod.html) | FSK | 2 tones; phase resets at each symbol boundary |
| 4 | [4FSK](https://www.mathworks.com/help/comm/ref/fskmod.html) | FSK | 4 tones; phase resets at each symbol boundary |
| 5 | [CPFSK](https://www.mathworks.com/help/comm/ref/comm.cpmmodulator-system-object.html) | Continuous-phase FSK | Binary rectangular frequency pulse; phase accumulated across symbols |
| 6 | [GFSK](https://www.mathworks.com/help/comm/ref/comm.cpmmodulator-system-object.html) | Gaussian FSK | Binary frequency sequence smoothed by a Gaussian kernel before phase integration |
| 7 | [PAM4](https://www.mathworks.com/help/comm/ref/pammod.html) | PAM | Levels `[-3,-1,1,3]/sqrt(5)` |
| 8 | [QAM16](https://www.mathworks.com/help/comm/ref/qammod.html) | QAM | Normalized square `4 x 4` constellation |
| 9 | [QAM64](https://www.mathworks.com/help/comm/ref/qammod.html) | QAM | Normalized square `8 x 8` constellation |
| 10 | [QAM256](https://www.mathworks.com/help/comm/ref/qammod.html) | QAM | Normalized square `16 x 16` constellation |
| 11 | [AM-DSB](https://www.mathworks.com/help/comm/ug/analog-passband-modulation.html) | Analog AM | Real complex-baseband envelope of a randomized two-tone message |
| 12 | [AM-SSB](https://www.mathworks.com/help/comm/ref/ssbmod.html) | Analog AM | Analytic signal of the randomized two-tone message |
| 13 | [WBFM](https://www.mathworks.com/help/comm/ref/fmmod.html) | Analog FM | Message-integrated phase with deviation parameter 4.0 |
| 14 | [LFM](https://www.mathworks.com/help/signal/ref/chirp.html) | Linear chirp | Random up/down sweep, center frequency and bandwidth, with a Hamming window |

PSK, PAM, and QAM use root-raised-cosine pulse shaping with roll-off 0.35
and span 6 symbols. FSK tones are equally spaced between -0.35 and 0.35
cycles/sample. The three FSK behaviors are intentionally distinct:

```text
2FSK/4FSK = discontinuous phase + rectangular frequency pulse
CPFSK     = continuous phase + rectangular frequency pulse
GFSK      = continuous phase + Gaussian-smoothed frequency pulse
```

LFM uses a center frequency sampled from `[-0.05, 0.05]` cycles/sample and a
sweep bandwidth sampled from `[0.50, 0.70]` cycles/sample. Sweep direction is
random. These limits keep the instantaneous frequency inside the normalized
Nyquist interval.

## Channel Dictionary

| ID (`channel`) | Channel | Meaning | Fixed definition |
|---:|---|---|---|
| 0 | [AWGN](https://www.mathworks.com/help/comm/ref/awgn.html) | Noise-only baseline | No fading filter before AWGN |
| 1 | [Rayleigh](https://www.mathworks.com/help/comm/ref/comm.rayleighchannel-system-object.html) | NLOS multipath fading | 5 complex taps over delays 0-4, exponential decay 1.5 |
| 2 | [Rician](https://www.mathworks.com/help/comm/ref/comm.ricianchannel-system-object.html) | Multipath with LOS | Rayleigh taps plus first-tap LOS, linear K-factor 6 |
| 3 | [Nakagami](https://www.mathworks.com/help/comm/ug/fading-channels.html) | Flexible fading severity | 5 taps, shape `m=2`, random phase, exponential decay |
| 4 | [DoublySelective](https://www.mathworks.com/help/comm/ug/fading-channels.html) | Time- and frequency-selective fading | Rayleigh multipath plus time-varying gain and Doppler phase |
| 5 | [FrequencySelectiveRayleigh](https://www.mathworks.com/help/comm/ref/comm.rayleighchannel-system-object.html) | Sparse delayed Rayleigh multipath | 6 taps over delays 0-10, exponential power decay 2.0 |
| 6 | [FrequencySelectiveRician](https://www.mathworks.com/help/comm/ref/comm.ricianchannel-system-object.html) | Sparse delayed Rician multipath | Frequency-selective Rayleigh plus first-tap LOS, linear K-factor 6 |
| 7 | [LognormalShadowing](https://www.mathworks.com/help/comm/propagation-and-channel-models.html) | Correlated multiplicative shadowing | 4 dB standard deviation, correlation length 24 samples |
| 8 | [3GPP_TDL_A](https://www.mathworks.com/help/5g/ref/nrtdlchannel-system-object.html) | 3GPP TR 38.901 TDL-A NLOS profile | 1 us delay spread, 2000 Hz maximum Doppler, 1 MHz SISO |
| 9 | [3GPP_TDL_C](https://www.mathworks.com/help/5g/ref/nrtdlchannel-system-object.html) | 3GPP TR 38.901 TDL-C NLOS profile | 1 us delay spread, 2000 Hz maximum Doppler, 1 MHz SISO |
| 10 | [3GPP_TDL_E](https://www.mathworks.com/help/5g/ref/nrtdlchannel-system-object.html) | 3GPP TR 38.901 TDL-E LOS profile | 1 us delay spread, 2000 Hz maximum Doppler, 1 MHz SISO |

The generator creates one reusable `nrTDLChannel` object per TDL profile and
resets its filter state between samples. This avoids constructing hundreds of
thousands of System objects while keeping samples separated.

Every sample follows the same processing order:

```text
clean modulation
-> selected fading/channel model
-> random CFO and carrier phase
-> complex AWGN at the requested SNR
-> RMS normalization
```

CFO is uniform in `[-0.01, 0.01]` cycles/sample and carrier phase is uniform in
`[0, 2*pi)`. AWGN power is calculated after the selected channel. Final RMS
normalization scales signal and noise together, so it does not alter SNR.

## Generate the Dataset

Run from MATLAB:

```matlab
cd <project>/src/datsetgenarator_matlab
run_generate_dataset
```

Equivalent function call:

```matlab
cfg = amr_dataset_config();
generate_amr_dataset(cfg);
```

Generation is disk-backed. The generator buffers one modulation-channel-SNR
cell at a time and writes it through `matfile`; it does not hold the full
dataset tensor in RAM.

The formal config has `overwriteExisting=true`. Starting generation removes an
existing output file at the same path. Set it to `false` before calling the
function when replacement should be prohibited.

The file stores:

```text
generationComplete = false
generatedSamples    = number written so far
```

Only after all cells and final validation succeed is `generationComplete` set
to `true`. If MATLAB is interrupted, the partial file remains marked incomplete
and both Python loading and MATLAB visualization reject it. Run generation
again to replace the incomplete file.

## Visualize the Dataset

After successful generation:

```matlab
cd <project>/src/datsetgenarator_matlab
run_visualize_dataset
```

Figures are written to:

```text
<project>/src/datsetgenarator_matlab/runs/matlab_dataset_viz
```

One PNG is created for every channel:

```text
iq_time_examples_AWGN.png
iq_time_examples_Rayleigh.png
...
iq_time_examples_3GPP_TDL_E.png
```

Each figure uses five panels per row, giving a 3-by-5 layout for the 15
modulation classes at 10 dB. I is blue and Q is green. There is no global
title, legend, or in-panel I/Q text; the channel is encoded in the file name.
Each panel title follows `modulation value-dB SNR`, for example
`BPSK 10dB SNR`. All panels in one figure use
the same symmetric amplitude limits. The visualizer calculates sample indices
from `meta.sampleOrder` and reads only the 15 required waveforms instead of
scanning or loading all 3.3 million samples.

Optional direct usage:

```matlab
% Show the first 8 classes at 0 dB for selected channels only.
visualize_amr_dataset(datasetPath, outputDir, 8, 0, ...
    {'AWGN', '3GPP_TDL_C'});
```

## Output Contract

| Variable | MATLAB type and shape | Meaning |
|---|---|---|
| `X` | `single [N,2,512]` | I/Q samples; index 1 is I, index 2 is Q |
| `y` | `int64 [N,1]` | Zero-based modulation ID |
| `snr` | `int64 [N,1]` | SNR in dB |
| `channel` | `int64 [N,1]` | Zero-based channel ID |
| `split` | `int8 [N,1]` | `0=train`, `1=validation`, `2=test` |
| `classNames` | cell array | Modulation names in label order |
| `channelNames` | cell array | Channel names in label order |
| `splitNames` | cell array | `train`, `val`, `test` |
| `meta` | struct | Complete generation configuration and ordering |
| `generatedSamples` | `int64` scalar | Number of samples written |
| `generationComplete` | logical scalar | Safe-to-use completion flag |

Labels are zero-based for direct use with PyTorch `CrossEntropyLoss`. Samples
are stored in this deterministic cell order:

```text
class -> channel -> SNR -> sample
```

The sample order is deterministic, but the train/validation/test assignment is
randomized independently inside every cell using seed 42.

## Split and Balance

| Item | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Each modulation-channel-SNR cell | 1000 | 600 | 200 | 200 |
| Whole dataset | 3,300,000 | 1,980,000 | 660,000 | 660,000 |

Additional deterministic totals:

| Unit | Samples |
|---|---:|
| Each modulation-channel pair | 20,000 |
| Each modulation-SNR pair | 11,000 |
| Each channel-SNR pair | 15,000 |
| Each modulation class | 220,000 |
| Each channel condition | 300,000 |
| Each SNR point | 165,000 |
