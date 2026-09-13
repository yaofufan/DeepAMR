%RUN_GENERATE_DATASET Generate the fixed formal MATLAB AMR dataset.

clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
cfg = amr_dataset_config();
numSamples = numel(cfg.classNames) * numel(cfg.channelNames) * ...
    numel(cfg.snrs) * cfg.samplesPerClassPerSnrPerChannel;

fprintf('Dataset output: %s\n', cfg.outputPath);
fprintf('Classes=%d, channels=%d, SNR points=%d, samples/cell=%d, length=%d\n', ...
    numel(cfg.classNames), numel(cfg.channelNames), numel(cfg.snrs), ...
    cfg.samplesPerClassPerSnrPerChannel, cfg.signalLength);
fprintf('Total samples=%d\n', numSamples);
fprintf('5G Toolbox nrTDLChannel available: %d\n', ...
    exist('nrTDLChannel', 'class') == 8 || exist('nrTDLChannel', 'file') == 2);
if exist(cfg.outputPath, 'file') && cfg.overwriteExisting
    fprintf('Existing output will be replaced after generation starts.\n');
end

generate_amr_dataset(cfg);
