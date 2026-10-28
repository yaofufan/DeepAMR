%RUN_PLOT_CHANNEL_COMPARE Compare all ten algorithms on an 11-channel radar.
clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
runsDir = fullfile(scriptDir, '..', 'runs');

% Same nine baselines and display names as run_plot_snr_compare.m.
evalDirs = {
    'eval_oshea_resnet'
    'eval_mcldnn'
    'eval_cctl_net'
    'eval_se_tfa'
    'eval_amcnet'
    'eval_mcformer'
    'eval_mccfn'
    'eval_tldnn'
    'eval_dhn'
    'eval_saftformer_full'
};
experiment.modelNames = {
    'ResNet', 'MCLDNN', 'CCTL-Net', 'SE-TFA', 'AMCNet', ...
    'MCformer', 'MCCFN', 'TLDNN', 'DHN', 'SAFTFormer (Ours)'
};
experiment.metricsFiles = cellfun(@(name) ...
    fullfile(runsDir, name, 'channel_accuracy.csv'), evalDirs, ...
    'UniformOutput', false);
% Current Full best.pt evaluation, epoch 41; matches the SNR comparison.
experiment.channelOrder = {
    'AWGN', 'Rayleigh', 'Rician', 'Nakagami', 'DoublySelective', ...
    'FrequencySelectiveRayleigh', 'FrequencySelectiveRician', ...
    'LognormalShadowing', '3GPP_TDL_A', '3GPP_TDL_C', '3GPP_TDL_E'
};
experiment.channelLabels = {
    'AWGN', 'Rayleigh', 'Rician', 'Nakagami', 'Doubly selective', ...
    sprintf('Frequency-selective\nRayleigh'), ...
    sprintf('Frequency-selective\nRician'), 'Lognormal shadowing', ...
    '3GPP TDL-A', '3GPP TDL-C', '3GPP TDL-E'
};
experiment.figureSize = [960, 800];
% Enlarge the circle inside the same canvas while retaining label/legend room.
experiment.axesPosition = [0.04, 0.16, 0.92, 0.82];
experiment.yLimits = [-1.10, 1.15];
experiment.legendBottom = 0.035;
experiment.rangeLabelOffset = 0.002;
experiment.bottomLabelRadius = 1.13;
% Separate the two multiline labels below the circle without shrinking text.
experiment.channelLabelOffsets = zeros(numel(experiment.channelOrder), 2);
experiment.channelLabelOffsets(6:7, 1) = [0.16; -0.16];
experiment.channelLabelOffsets(6:7, 2) = -0.10;
% Same point sizes and PDF scale as the SNR comparison.
experiment.fontSize = 14;
experiment.detailFontSize = 12;
experiment.legendFontSize = 14;
experiment.legendItemTokenSize = [16, 10];
% Paper-style per-channel bounds are computed from all ten algorithms.
% Each label shows (lower accuracy, upper accuracy) in percent.
% No chart title or footnote is drawn.
experiment.markers = {'o', 's', '^', 'd', 'v', 'p', 'h', '>', '<', 'o'};
experiment.lineStyles = {'-', '--', ':', '-.', '-', '--', ':', '-.', '-', '-'};
experiment.lineWidth = 1.1;
experiment.markerSize = 3.5;
experiment.spokeOpacity = 1.0;
experiment.spokeLineStyle = '--';
% Same ten-color palette and model order as the SNR comparison.
experiment.colors = [
     68, 119, 170
    238, 119,  51
     44, 146,  91
    173, 123, 164
     51, 187, 238
    204, 187,  68
     68, 170, 153
    123, 180, 125
    170,  68, 153
    238, 102, 119
] / 255;
experiment.exportFormats = {'pdf', 'fig'};

outputPath = fullfile(scriptDir, 'figures', 'channel_accuracy_radar.png');
[fig, comparisonTable] = plot_channel_accuracy_radar(experiment, outputPath);
disp(comparisonTable);
