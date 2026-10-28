%RUN_PLOT_DATASET_OVERVIEW Show fixed held-out examples of all 15 modulations.
clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
datasetPath = fullfile(scriptDir, '..', '..', 'datasets', 'matlab_amr_dataset.mat');
outputPath = fullfile(scriptDir, 'figures', 'dataset_modulation_overview.png');
[fig, provenance] = plot_dataset_overview(datasetPath, outputPath);

validationDir = fullfile(scriptDir, '..', 'tmp', 'dataset_overview_20261026');
if ~exist(validationDir, 'dir')
    mkdir(validationDir);
end
manifestPath = fullfile(validationDir, 'dataset_overview_provenance.json');
manifestFile = fopen(manifestPath, 'w', 'n', 'UTF-8');
if manifestFile < 0
    error('Cannot write provenance manifest: %s.', manifestPath);
end
manifestCleanup = onCleanup(@() fclose(manifestFile));
fprintf(manifestFile, '%s\n', jsonencode(provenance, 'PrettyPrint', true));
clear manifestCleanup;
fprintf('Saved dataset overview: %s\n', outputPath);
fprintf('MATLAB sample IDs: %s\n', mat2str([provenance.examples.matlabSampleIndex]));
fprintf('Selection: first stored test example per modulation, AWGN, 18 dB.\n');
