%RUN_VISUALIZE_DATASET Create paper-style I/Q time-domain example figures.

clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
cfg = amr_dataset_config();
outputDir = fullfile(scriptDir, 'runs', 'matlab_dataset_viz');
fprintf('Dataset input: %s\n', cfg.outputPath);
fprintf('Figure output: %s\n', outputDir);
visualize_amr_dataset(cfg.outputPath, outputDir);
