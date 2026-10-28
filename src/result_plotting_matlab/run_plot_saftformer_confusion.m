%RUN_PLOT_SAFTFORMER_CONFUSION Overall, 0 dB and 10 dB count matrices.
% Reuses the formal Full best.pt predictions, without another evaluation.
clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
evalDir = fullfile(scriptDir, '..', 'runs', 'eval_saftformer_full');
csvFile = fullfile(evalDir, 'confusion_matrix.csv');
predictionsFile = fullfile(evalDir, 'predictions.mat');
provenanceFile = fullfile(evalDir, 'evaluation_provenance.json');
provenance = jsondecode(fileread(provenanceFile));
assert(strcmp(provenance.checkpoint_selection, 'best.pt'), ...
    'The formal comparison must use the recorded best.pt checkpoint.');
assert(strcmp(provenance.split, 'test') && provenance.test_samples == 660000, ...
    'Expected the formal 660,000-example test evaluation.');

raw = readcell(csvFile);
classNames = string(raw(1, 2:end))';
csvCounts = cell2mat(raw(2:end, 2:end));
assert(isequal(classNames, string(raw(2:end, 1))), ...
    'CSV row and column class order must match.');
numClasses = numel(classNames);
assert(numClasses == 15);
predictions = load(predictionsFile, 'y_true', 'y_pred', 'snr', 'channel', 'classNames');
yTrue = double(predictions.y_true(:));
yPred = double(predictions.y_pred(:));
snr = double(predictions.snr(:));
channel = double(predictions.channel(:));
assert(numel(yTrue) == provenance.test_samples ...
    && numel(yPred) == numel(yTrue) && numel(snr) == numel(yTrue) ...
    && numel(channel) == numel(yTrue));
assert(isequal(string(predictions.classNames(:)), classNames));
validateattributes(yTrue, {'numeric'}, {'integer', '>=', 0, '<', numClasses});
validateattributes(yPred, {'numeric'}, {'integer', '>=', 0, '<', numClasses});
assert(isequal(unique(snr), (-20:2:18)'));
assert(isequal(unique(channel), (0:10)'));
panelIndices = {find(true(size(yTrue))), find(snr == 0), find(snr == 10)};
counts = zeros(numClasses, numClasses, 3);
expectedSamples = [660000, 33000, 33000];
expectedRowSamples = [44000, 2200, 2200];
for panel = 1:3
    selected = panelIndices{panel};
    counts(:, :, panel) = accumarray([yTrue(selected) + 1, yPred(selected) + 1], ...
        1, [numClasses, numClasses]);
    assert(numel(selected) == expectedSamples(panel));
    assert(all(sum(counts(:, :, panel), 2) == expectedRowSamples(panel)));
    panelIndices{panel} = uint32(selected);
end
assert(isequal(counts(:, :, 1), csvCounts), ...
    'Overall predictions must reproduce confusion_matrix.csv exactly.');

options.FigureSize = [1880, 720];
options.FontSize = 13;
options.CellFontSize = 9.5;
options.AxisLabelFontSize = 14;
options.TitleFontSize = 20;
options.PanelNames = {'Overall', '0 dB', '10 dB'};
options.ExportFormats = {'pdf', 'fig'};
options.Source = struct('predictionsMat', predictionsFile, ...
    'overallConfusionCsv', csvFile, 'evaluationProvenanceFile', provenanceFile);
options.Provenance = provenance;
options.PanelIndices = panelIndices;
options.Conditions = struct('label', {'Overall', '0 dB', '10 dB'}, ...
    'snrLevelsDb', {-20:2:18, 0, 10}, 'channelCount', {11, 11, 11});
outputPath = fullfile(scriptDir, 'figures', 'saftformer_confusion_triptych.png');
[fig, matrixData] = plot_confusion_triptych(counts, classNames, outputPath, options);
assert(isequal(matrixData.panelSampleCounts, expectedSamples));
assert(isequal(matrixData.displayValues, counts) && strcmp(matrixData.normalization, 'none'));
fprintf('Verified count matrices: Overall=%d, 0dB=%d, 10dB=%d.\n', expectedSamples);
