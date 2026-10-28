%RUN_PLOT_SNR_COMPARE Compare the current best model with evaluated baselines.
clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
runsDir = fullfile(scriptDir, '..', 'runs');

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
modelNames = {
    'ResNet', 'MCLDNN', 'CCTL-Net', 'SE-TFA', 'AMCNet', ...
    'MCformer', 'MCCFN', 'TLDNN', 'DHN', 'SAFTFormer (Ours)'
};

% Ours uses the current Full best.pt evaluation (epoch 41; 69.808485% accuracy),
% matching the trained-checkpoint benchmark comparison and presentation table.
% Keep one curve per comparison algorithm.
% exp(1).title = 'Automatic Modulation Recognition';
exp(1).metricsFiles = cellfun(@(name) ...
    fullfile(runsDir, name, 'snr_accuracy.csv'), evalDirs, 'UniformOutput', false);
exp(1).modelNames = modelNames;
exp(1).figureSize = [960, 800];
% Matched presentation typography in both comparison figures.
exp(1).fontSize = 14;
exp(1).axisLabelFontSize = 14;
exp(1).detailFontSize = 12;
exp(1).legendFontSize = 14;
exp(1).legendItemTokenSize = [16, 10];
exp(1).squareAxes = false;
exp(1).filledMarkers = true;
exp(1).tickDirection = 'in';
exp(1).xPadding = 0.5;
exp(1).markers = {'o', 's', '^', 'd', 'v', 'p', 'h', '>', '<', 'o'};
exp(1).accuracyPercent = true;
exp(1).strictSNRGrid = true;
exp(1).showPanelLabel = false;
exp(1).legendLocation = 'northwest';
exp(1).legendNumColumns = 2;
exp(1).highlightModelIndex = numel(modelNames);
% Adapted from Paul Tol's bright, vibrant, muted, and banded schemes.
% This ten-color selection is a custom combination, not a canonical scheme.
% Source: https://sronpersonalpages.nl/~pault/
exp(1).colors = [
     68, 119, 170  % #4477AA - O'Shea ResNet: blue
    238, 119,  51  % #EE7733 - MCLDNN: orange
     44, 146,  91  % #2C925B - CCTL-Net: green
    173, 123, 164  % #AD7BA4 - SE-TFA: lavender
     51, 187, 238  % #33BBEE - AMCNet: cyan
    204, 187,  68  % #CCBB44 - MCformer: gold
     68, 170, 153  % #44AA99 - MCCFN: teal
    123, 180, 125  % #7BB47D - TLDNN: light green
    170,  68, 153  % #AA4499 - DHN: purple
    238, 102, 119  % #EE6677 - Ours: coral red
] / 255;
exp(1).insetXLim = [0, 18];
exp(1).insetYLim = [0.79, 0.95];
exp(1).connectInset = true;
exp(1).exportFormats = {'pdf', 'fig'};

outputPath = fullfile(scriptDir, 'figures', 'snr_accuracy_compare.png');
plot_snr_accuracy_paper(exp, outputPath);
