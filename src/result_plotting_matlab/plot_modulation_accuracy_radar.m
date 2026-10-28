function [fig, data] = plot_modulation_accuracy_radar(csvFile, outputPath, options)
%PLOT_MODULATION_ACCURACY_RADAR Plot one model's mean accuracy by modulation.
% Each spoke is the measured mean over all twenty -20:2:18 dB levels.
% The CSV already averages the channel conditions within each class/SNR cell.
% All spokes use one absolute 0--100% radial scale; no per-spoke rescaling,
% interpolation, smoothing, or area fill is applied. The figure has no title
% or legend. PNG, vector PDF and editable FIG retain exact data/provenance.

arguments
    csvFile (1, :) char
    outputPath (1, :) char
    options struct = struct()
end

if ~isfile(csvFile)
    error('Result file not found: %s', csvFile);
end
sourceTable = readtable(csvFile, 'TextType', 'string');
if ~all(ismember({'modulation', 'snr', 'accuracy'}, sourceTable.Properties.VariableNames))
    error('CSV must contain modulation, snr, and accuracy columns.');
end
if isempty(sourceTable) ...
        || any(ismissing(sourceTable.modulation) | strlength(sourceTable.modulation) == 0)
    error('Modulation labels must be nonempty.');
end
validateattributes(sourceTable.snr, {'numeric'}, {'real', 'finite', 'vector'});
validateattributes(sourceTable.accuracy, {'numeric'}, ...
    {'real', 'finite', 'vector', '>=', 0, '<=', 1});
classNames = string(option(options, 'ClassOrder', ...
    unique(sourceTable.modulation, 'stable')));
classNames = classNames(:);
numClasses = numel(classNames);
if numClasses < 3 || numel(unique(classNames)) ~= numClasses ...
        || ~isequal(sort(classNames), sort(unique(sourceTable.modulation)))
    error('ClassOrder must contain every measured class exactly once, with at least three classes.');
end
snr = (-20:2:18)';
accuracyFractions = zeros(numClasses, numel(snr));
for classIdx = 1:numClasses
    rows = sourceTable(sourceTable.modulation == classNames(classIdx), :);
    [measuredSNR, order] = sort(rows.snr);
    if ~isequal(measuredSNR(:), snr)
        error('Each class must contain exactly one row per -20:2:18 dB level: %s.', ...
            classNames(classIdx));
    end
    accuracyFractions(classIdx, :) = rows.accuracy(order)';
end
meanAccuracyPercent = 100 * mean(accuracyFractions, 2);
normalizedRadius = meanAccuracyPercent / 100;

confusionFile = char(option(options, 'ConfusionMatrixFile', ''));
confusionCounts = [];
if ~isempty(confusionFile)
    if ~isfile(confusionFile)
        error('Confusion matrix not found: %s', confusionFile);
    end
    raw = readcell(confusionFile);
    rowNames = string(raw(2:end, 1));
    columnNames = string(raw(1, 2:end))';
    confusionCounts = cell2mat(raw(2:end, 2:end));
    if ~isequal(rowNames(:), classNames) || ~isequal(columnNames(:), classNames)
        error('Confusion-matrix class order does not match the radar.');
    end
    validateattributes(confusionCounts, {'numeric'}, ...
        {'real', 'finite', 'nonnegative', 'integer', 'size', [numClasses, numClasses]});
    support = sum(confusionCounts, 2);
    if any(support <= 0) ...
            || any(abs(diag(confusionCounts) ./ support - normalizedRadius) > 1e-10)
        error('The measured SNR means disagree with confusion-matrix class accuracy.');
    end
end
provenanceFile = char(option(options, 'ProvenanceFile', ''));
provenance = struct();
if ~isempty(provenanceFile)
    if ~isfile(provenanceFile)
        error('Evaluation provenance not found: %s', provenanceFile);
    end
    provenance = jsondecode(fileread(provenanceFile));
    if ~isfield(provenance, 'checkpoint_selection') ...
            || ~strcmp(provenance.checkpoint_selection, 'best.pt') ...
            || ~isfield(provenance, 'split') || ~strcmp(provenance.split, 'test')
        error('Expected the formal best.pt checkpoint on the test split.');
    end
    if ~isempty(confusionCounts) && (~isfield(provenance, 'test_samples') ...
            || provenance.test_samples ~= sum(confusionCounts, 'all'))
        error('The confusion matrix and provenance have different test sample counts.');
    end
end

figureSize = option(options, 'FigureSize', [1000, 800]);
fontSize = option(options, 'FontSize', 18);
radialFontSize = option(options, 'RadialFontSize', 14);
lineWidth = option(options, 'LineWidth', 1.5);
markerSize = option(options, 'MarkerSize', 4.5);
color = option(options, 'Color', [238, 102, 119] / 255);
modelName = char(option(options, 'ModelName', 'SAFTFormer (Ours)'));
validateattributes(figureSize, {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 2});
validateattributes([fontSize, radialFontSize, lineWidth, markerSize], {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 4});
validateattributes(color, {'numeric'}, ...
    {'real', 'finite', 'size', [1, 3], '>=', 0, '<=', 1});

fig = figure('Color', 'white', 'Units', 'pixels', ...
    'Position', [80, 60, figureSize(:)'], 'Renderer', 'painters');
ax = axes(fig, 'Units', 'normalized', 'Position', [0.06, 0.035, 0.88, 0.91], ...
    'XLim', [-1.42, 1.42], 'YLim', [-1.20, 1.20], ...
    'DataAspectRatio', [1, 1, 1], 'Visible', 'off', ...
    'Tag', 'modulation_radar_axes');
hold(ax, 'on');
theta = pi / 2 - (0:numClasses - 1)' * 2 * pi / numClasses;
circleTheta = linspace(0, 2 * pi, 361);
guideColor = [0.72, 0.72, 0.72];
for radialTick = 20:20:80
    r = radialTick / 100;
    plot(ax, r * cos(circleTheta), r * sin(circleTheta), '--', ...
        'Color', guideColor, 'LineWidth', 0.55, ...
        'HandleVisibility', 'off', 'Tag', 'modulation_radar_ring');
end
for classIdx = 1:numClasses
    line(ax, [0, cos(theta(classIdx))], [0, sin(theta(classIdx))], ...
        'Color', guideColor, 'LineWidth', 0.55, 'LineStyle', '--', ...
        'HandleVisibility', 'off', 'Tag', 'modulation_radar_spoke');
end
plot(ax, cos(circleTheta), sin(circleTheta), '-', 'Color', 'black', ...
    'LineWidth', 0.8, 'HandleVisibility', 'off', ...
    'Tag', 'modulation_radar_outer_circle');

x = normalizedRadius .* cos(theta);
y = normalizedRadius .* sin(theta);
plot(ax, [x; x(1)], [y; y(1)], '-', 'Color', color, ...
    'LineWidth', lineWidth, 'Marker', 'o', 'MarkerSize', markerSize, ...
    'MarkerIndices', 1:numClasses, 'MarkerFaceColor', color, ...
    'MarkerEdgeColor', color, 'DisplayName', modelName, ...
    'Tag', 'modulation_radar_curve');
for classIdx = 1:numClasses
    cosine = cos(theta(classIdx));
    if cosine > 0.35
        alignment = 'left';
    elseif cosine < -0.35
        alignment = 'right';
    else
        alignment = 'center';
    end
    text(ax, 1.10 * cosine, 1.10 * sin(theta(classIdx)), classNames(classIdx), ...
        'HorizontalAlignment', alignment, 'VerticalAlignment', 'middle', ...
        'FontName', 'Times New Roman', 'FontSize', fontSize, 'Color', 'black', ...
        'Interpreter', 'none', 'Clipping', 'off', ...
        'Tag', 'modulation_radar_class_label');
end
% Absolute tick labels sit just right of the upper spoke, inside each ring.
for radialTick = 20:20:100
    text(ax, 0.025, radialTick / 100, sprintf('%d%%', radialTick), ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
        'FontName', 'Times New Roman', 'FontSize', radialFontSize, ...
        'Color', 'black', 'Interpreter', 'none', 'Clipping', 'off', ...
        'Tag', 'modulation_radar_radial_label');
end

data = struct('modelName', modelName, 'classNames', classNames, 'snr', snr, ...
    'accuracyFractions', accuracyFractions, ...
    'meanAccuracyPercent', meanAccuracyPercent, 'normalizedRadius', normalizedRadius, ...
    'radialLimitsPercent', [0, 100], 'radialTicksPercent', 20:20:100, ...
    'theta', theta, 'sourceFile', csvFile, 'sourceTable', sourceTable, ...
    'confusionMatrixFile', confusionFile, 'confusionCounts', confusionCounts, ...
    'provenanceFile', provenanceFile, 'provenance', provenance, ...
    'meanDefinition', 'Mean over all 20 equally populated -20:2:18 dB levels.', ...
    'independentSpokeScaling', false, 'smoothingApplied', false);
fig.UserData = data;
drawnow;
[outDir, baseName] = fileparts(outputPath);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end
exportgraphics(fig, outputPath, 'Resolution', 300, 'BackgroundColor', 'white');
exportgraphics(fig, fullfile(outDir, [baseName, '.pdf']), ...
    'ContentType', 'vector', 'BackgroundColor', 'white');
savefig(fig, fullfile(outDir, [baseName, '.fig']));
fprintf('Saved mean modulation radar: %s (.png, .pdf, .fig).\n', fullfile(outDir, baseName));
end

function value = option(options, name, default)
if isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default;
end
end
