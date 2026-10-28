%RUN_PLOT_SAFTFORMER_MODULES Inspect one predefined Full best.pt test example.
% First run from src: python -m amc.export_layer_visualizations
clc;
clear;
close all;
scriptDir=fileparts(mfilename('fullpath'));
matFile=fullfile(scriptDir,'feature_data','saftformer_layer_visualizations.mat');
options.SampleIndex=1; % First seeded BPSK / AWGN / 0 dB example, not chosen by its response.
options.FigureSize=[1320,650];
options.FontSize=15;
outputPath=fullfile(scriptDir,'figures','saftformer_module_responses.png');
[fig,moduleData]=plot_saftformer_module_responses(matFile,outputPath,options);
