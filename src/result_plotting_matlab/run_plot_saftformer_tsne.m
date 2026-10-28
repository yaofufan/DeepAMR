%RUN_PLOT_SAFTFORMER_TSNE Three fixed-seed independent feature projections.
% Feature extraction uses the same Full checkpoint and balanced test subsets.
% Re-running after layout changes reuses a hash-checked embedding cache.
clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
featureDir = fullfile(scriptDir, 'feature_data');
featureFiles = {
    fullfile(featureDir, 'saftformer_features_overall.mat')
    fullfile(featureDir, 'saftformer_features_0dB.mat')
    fullfile(featureDir, 'saftformer_features_10dB.mat')
};
options.PanelTitles = {'Overall', '0 dB', '10 dB'};
options.Seed = 42;
options.Perplexity = 30;
options.ExpectedFeatureDimensions = 48;
options.ExpectedSamplesPerClass = 300;
options.FigureSize = [1600, 760];
options.MarkerSize = 8;
options.TitleFontSize = 22;
options.LegendFontSize = 18;
options.CacheFile = fullfile(scriptDir, 'figure_data', 'saftformer_tsne_embeddings.mat');
outputPath = fullfile(scriptDir, 'figures', 'saftformer_tsne_triptych.png');
[fig, triptychData] = plot_tsne_triptych(featureFiles, outputPath, options);
fprintf('Projection cache reused: %s\n', mat2str(triptychData.cacheReused'));
