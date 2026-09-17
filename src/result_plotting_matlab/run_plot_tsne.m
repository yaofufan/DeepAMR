clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fullfile(scriptDir, '..', '..');
featureFile = fullfile(projectRoot, 'src', 'runs', ...
    'eval_oshea_resnet', 'tsne_features_0dB.mat');
outputPath = fullfile(scriptDir, 'figures', 'tsne_oshea_resnet_0dB.png');

options.Seed = 42;
options.Perplexity = 30;
options.MarkerSize = 12;
plot_tsne_features(featureFile, outputPath, options);
