%RUN_PLOT_DATASET_EXAMPLES Plot fixed held-out QPSK examples from the dataset.
clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
datasetPath = fullfile(scriptDir, '..', '..', 'datasets', 'matlab_amr_dataset.mat');
outputPath = fullfile(scriptDir, 'figures', 'dataset_signal_examples.png');
[fig, provenance] = plot_dataset_examples(datasetPath, outputPath);

validationDir = fullfile(scriptDir, '..', 'tmp', 'dataset_slides_20261026');
if ~exist(validationDir, 'dir')
    mkdir(validationDir);
end
manifestPath = fullfile(validationDir, 'dataset_examples_provenance.json');
manifestFile = fopen(manifestPath, 'w', 'n', 'UTF-8');
if manifestFile < 0
    error('Cannot write provenance manifest: %s.', manifestPath);
end
manifestCleanup = onCleanup(@() fclose(manifestFile));
fprintf(manifestFile, '%s\n', jsonencode(provenance, 'PrettyPrint', true));
clear manifestCleanup;
fprintf('Saved dataset examples: %s\n', outputPath);
fprintf('MATLAB sample IDs: %s\n', mat2str([provenance.examples.matlabSampleIndex]));
fprintf('Selection: first stored test example within each requested cell.\n');
