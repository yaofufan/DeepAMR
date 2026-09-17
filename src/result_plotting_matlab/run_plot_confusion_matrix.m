clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fullfile(scriptDir, '..', '..');
csvFile = fullfile(projectRoot, 'src', 'runs', ...
    'eval_oshea_resnet', 'confusion_matrix.csv');
outputPath = fullfile(scriptDir, 'figures', 'confusion_matrix.png');

options.Title = 'Confusion Matrix';
plot_confusion_matrix_results(csvFile, outputPath, options);
