clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fullfile(scriptDir, '..', '..');
csvFile = fullfile(projectRoot, 'src', 'runs', ...
    'eval_oshea_resnet', 'modulation_snr_accuracy.csv');
outputPath = fullfile(scriptDir, 'figures', 'modulation_snr_accuracy.png');

options.Title = 'Per-Modulation Accuracy vs SNR';
options.LegendColumns = 5;
plot_modulation_snr_accuracy(csvFile, outputPath, options);
