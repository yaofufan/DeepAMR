function [fig, curveData] = plot_modulation_snr_accuracy(csvFile, outputPath, options)
%PLOT_MODULATION_SNR_ACCURACY Plot the measured SNR curve of every class.
% Existing callers retain fraction-valued axes, the original class palette,
% title, southeast legend, and PNG-only export. AccuracyScale = 'percent'
% gives a 0--100 axis; an explicitly empty Title draws no title.
% ExportFormats = {'pdf','fig'} adds vector PDF and editable FIG output.
% XLimits/YLimits optionally pad the axes in displayed units without adding
% measurements. GridAlpha optionally matches an existing figure style.
% No interpolation or smoothing is performed. Returned curveData and FIG
% UserData retain all original points, display values, and source provenance.

arguments
    csvFile (1, :) char
    outputPath (1, :) char
    options struct = struct()
end

if ~isfile(csvFile)
    error('Result file not found: %s', csvFile);
end
data = readtable(csvFile, 'TextType', 'string');
if ~all(ismember({'modulation', 'snr', 'accuracy'}, data.Properties.VariableNames))
    error('CSV must contain modulation, snr, and accuracy columns.');
end
if isempty(data) || any(ismissing(data.modulation) | strlength(data.modulation) == 0)
    error('CSV modulation labels must be nonempty.');
end
validateattributes(data.snr, {'numeric'}, {'real', 'finite', 'vector'});
validateattributes(data.accuracy, {'numeric'}, ...
    {'real', 'finite', 'vector', '>=', 0, '<=', 1});
classNames = string(option(options, 'ClassOrder', unique(data.modulation, 'stable')));
classNames = classNames(:);
numClasses = numel(classNames);
if numel(unique(classNames)) ~= numClasses ...
        || ~isequal(sort(classNames), sort(unique(data.modulation)))
    error('ClassOrder must contain every measured modulation exactly once.');
end
snr = unique(data.snr, 'sorted');
expectedSNR = option(options, 'ExpectedSNR', snr);
if ~isequal(snr(:), expectedSNR(:))
    error('Measured SNR levels do not match ExpectedSNR.');
end
accuracyFractions = zeros(numClasses, numel(snr));
for classIdx = 1:numClasses
    rows = data(data.modulation == classNames(classIdx), :);
    [measuredSNR, order] = sort(rows.snr);
    if ~isequal(measuredSNR(:), snr(:))
        error('Each modulation must contain one row per SNR level: %s.', classNames(classIdx));
    end
    accuracyFractions(classIdx, :) = rows.accuracy(order)';
end

scale = validatestring(option(options, 'AccuracyScale', 'fraction'), {'fraction', 'percent'});
if strcmp(scale, 'percent')
    multiplier = 100;
    yLabel = 'Accuracy (%)';
else
    multiplier = 1;
    yLabel = 'Accuracy';
end
plottedAccuracy = multiplier * accuracyFractions;
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
end

figureSize = option(options, 'FigureSize', [900, 650]);
fontSize = option(options, 'FontSize', 10);
axisLabelFontSize = option(options, 'AxisLabelFontSize', fontSize);
legendFontSize = option(options, 'LegendFontSize', 7);
legendColumns = option(options, 'LegendColumns', 5);
colors = amr_class_colors(numClasses);
markers = {'o', 's', '^', 'd', 'v', '>', '<', 'p', 'h'};
lineWidth = option(options, 'LineWidth', 1.35);
markerSize = option(options, 'MarkerSize', 4);
xLimits = option(options, 'XLimits', [-20, 18]);
yLimits = option(options, 'YLimits', [0, multiplier]);
gridAlpha = option(options, 'GridAlpha', 0.35);
validateattributes(xLimits, {'numeric'}, {'real','finite','increasing','numel',2});
validateattributes(yLimits, {'numeric'}, {'real','finite','increasing','numel',2});
validateattributes(gridAlpha, {'numeric'}, {'real','finite','scalar','>=',0,'<=',1});
if min(snr) < xLimits(1) || max(snr) > xLimits(2) ...
        || min(plottedAccuracy, [], 'all') < yLimits(1) ...
        || max(plottedAccuracy, [], 'all') > yLimits(2)
    error('Axis limits must retain all measured data points.');
end
validateattributes(figureSize, {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 2});
validateattributes([fontSize, axisLabelFontSize, legendFontSize, lineWidth, markerSize], ...
    {'numeric'}, {'real', 'finite', 'positive', 'vector', 'numel', 5});

fig = figure('Color', 'white', 'Units', 'pixels', ...
    'Position', [100, 80, figureSize(:)'], 'Renderer', 'painters');
ax = axes(fig, 'Tag', 'modulation_snr_axes');
hold(ax, 'on');
curves = gobjects(numClasses, 1);
for classIdx = 1:numClasses
    curves(classIdx) = plot(ax, snr, plottedAccuracy(classIdx, :), ...
        'Color', colors(classIdx, :), ...
        'Marker', markers{mod(classIdx - 1, numel(markers)) + 1}, ...
        'MarkerFaceColor', colors(classIdx, :), ...
        'MarkerEdgeColor', colors(classIdx, :), ...
        'MarkerIndices', 1:numel(snr), 'LineWidth', lineWidth, ...
        'MarkerSize', markerSize, 'DisplayName', classNames(classIdx), ...
        'Tag', 'modulation_snr_curve');
end
set(ax, 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'GridLineStyle', ':', 'GridAlpha', gridAlpha, 'GridColor', 'black', ...
    'FontName', 'Times New Roman', 'FontSize', fontSize, ...
    'XColor', 'black', 'YColor', 'black', 'TickDir', 'in', ...
    'LineWidth', 0.8, 'Layer', 'top', 'TickLabelInterpreter', 'none');
xlabel(ax, 'SNR (dB)', 'FontName', 'Times New Roman', ...
    'FontSize', axisLabelFontSize, 'Color', 'black');
ylabel(ax, yLabel, 'FontName', 'Times New Roman', ...
    'FontSize', axisLabelFontSize, 'Color', 'black');
if isfield(options, 'Title')
    titleText = options.Title;
else
    titleText = 'Per-Modulation Accuracy vs SNR';
end
if ~isempty(titleText)
    title(ax, titleText, 'FontName', 'Times New Roman', ...
        'FontWeight', 'normal', 'Color', 'black', 'Interpreter', 'none');
end
xlim(ax, xLimits);
xticks(ax, -20:2:18);
xtickangle(ax, 0);
ylim(ax, yLimits);
yticks(ax, option(options, 'YTicks', (0:0.1:1) * multiplier));
lgd = legend(ax, curves, cellstr(classNames), ...
    'Location', option(options, 'LegendLocation', 'southeast'), ...
    'NumColumns', min(legendColumns, numClasses), ...
    'FontName', 'Times New Roman', 'FontSize', legendFontSize, ...
    'TextColor', 'black', 'Box', 'on', 'Interpreter', 'none', 'AutoUpdate', 'off');
lgd.ItemTokenSize = option(options, 'LegendItemTokenSize', [24, 12]);
if isfield(options, 'LegendPosition') && ~isempty(options.LegendPosition)
    lgd.Units = 'normalized';
    lgd.Position = options.LegendPosition;
end
if isfield(options, 'AxesPosition') && ~isempty(options.AxesPosition)
    ax.Units = 'normalized';
    ax.Position = options.AxesPosition;
end

curveData = struct('classNames', classNames, 'snr', snr, ...
    'accuracyFractions', accuracyFractions, 'plottedAccuracy', plottedAccuracy, ...
    'accuracyScale', scale, 'sourceFile', csvFile, 'sourceTable', data, ...
    'provenanceFile', provenanceFile, 'provenance', provenance, ...
    'colors', colors, 'markers', {markers}, 'lineWidth', lineWidth, ...
    'markerSize', markerSize, 'smoothingApplied', false, ...
    'xLimits', xLimits, 'yLimits', yLimits);
fig.UserData = curveData;
drawnow;
outputDir = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
exportgraphics(fig, outputPath, 'Resolution', 300, 'BackgroundColor', 'white');
[folder, baseName] = fileparts(outputPath);
formats = string(option(options, 'ExportFormats', {}));
for format = formats(:)'
    target = fullfile(folder, [baseName, '.', char(lower(format))]);
    switch lower(format)
        case "pdf"
            exportgraphics(fig, target, 'ContentType', 'vector', 'BackgroundColor', 'white');
        case "fig"
            savefig(fig, target);
        otherwise
            error('Unsupported export format: %s. Use pdf or fig.', format);
    end
end
fprintf('Saved per-modulation SNR figure to %s\n', outputPath);
end

function value = option(options, name, default)
if isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default;
end
end
