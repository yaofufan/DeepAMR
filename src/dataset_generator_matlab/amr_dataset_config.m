function cfg = amr_dataset_config()
%AMR_DATASET_CONFIG Fixed formal MATLAB AMR dataset configuration.
%
% This configuration is the single dataset setting used by the project:
%   15 modulation classes
%   11 channel conditions
%   SNR -20:2:18 dB
%   1000 samples per modulation-channel-SNR cell
%   train/val/test = 6:2:2 inside each cell
%   signal shape [2, 512]

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fullfile(scriptDir, '..', '..');

cfg.outputPath = fullfile(projectRoot, 'datasets', 'matlab_amr_dataset.mat');
cfg.seed = 42;
cfg.overwriteExisting = true;

cfg.classNames = { ...
    'BPSK', 'QPSK', '8PSK', ...
    '2FSK', '4FSK', 'CPFSK', 'GFSK', ...
    'PAM4', ...
    'QAM16', 'QAM64', 'QAM256', ...
    'AM-DSB', 'AM-SSB', 'WBFM', ...
    'LFM' ...
};

cfg.channelNames = { ...
    'AWGN', 'Rayleigh', 'Rician', 'Nakagami', 'DoublySelective', ...
    'FrequencySelectiveRayleigh', 'FrequencySelectiveRician', ...
    'LognormalShadowing', '3GPP_TDL_A', '3GPP_TDL_C', '3GPP_TDL_E' ...
};

cfg.snrs = -20:2:18;
cfg.samplesPerClassPerSnrPerChannel = 1000;

cfg.splitNames = {'train', 'val', 'test'};
cfg.splitFractions = [0.6, 0.2, 0.2];

cfg.signalLength = 512;
cfg.samplesPerSymbol = 8;
cfg.rrcRollOff = 0.35;
cfg.rrcSpan = 6;
cfg.useRrcIfAvailable = true;
cfg.fskNormalizedToneRange = [-0.35, 0.35];
cfg.gfskGaussianWidthSymbols = 0.6;
cfg.lfmCenterFrequencyRange = [-0.05, 0.05];
cfg.lfmSweepBandwidthRange = [0.50, 0.70];
cfg.wbfmDeviation = 4.0;

cfg.enableCfo = true;
cfg.maxCfoCyclesPerSample = 0.01;
cfg.enableRandomPhase = true;
cfg.maxMultipathDelay = 4;
cfg.multipathDecay = 1.5;
cfg.frequencySelectiveNumTaps = 6;
cfg.frequencySelectiveMaxDelay = 10;
cfg.frequencySelectiveDecay = 2.0;
cfg.ricianKFactor = 6;
cfg.nakagamiM = 2;
cfg.maxDopplerCyclesPerSample = 0.02;
cfg.doublySelectiveSegmentLength = 16;
cfg.lognormalShadowingSigmaDb = 4;
cfg.lognormalShadowingCorrelationLength = 24;

cfg.sampleRate = 1e6;
cfg.tdlDelaySpreadSeconds = 1e-6;
cfg.tdlMaximumDopplerHz = 2000;
cfg.require5GToolbox = true;

cfg.normalizeRms = true;
cfg.saveVersion = '-v7.3';
end
