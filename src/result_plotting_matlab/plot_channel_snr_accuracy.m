function [fig, curveData] = plot_channel_snr_accuracy(csvFile, outputPath, options)
%PLOT_CHANNEL_SNR_ACCURACY Plot one measured SNR-accuracy curve per channel.
% Inputs are fractional accuracies. All twenty -20:2:18 dB points must occur
% exactly once for each channel. Curves are displayed as percentages without
% smoothing or interpolation. Optional XLimits/YLimits leave marker headroom.
% Exports PNG, vector PDF and editable FIG; returned data and FIG UserData
% preserve the complete raw table, display values and evaluation provenance.

arguments
    csvFile (1, :) char
    outputPath (1, :) char
    options struct = struct()
end

if ~isfile(csvFile)
    error('Result file not found: %s', csvFile);
end
sourceTable = readtable(csvFile, 'TextType', 'string');
if ~all(ismember({'channel', 'snr', 'accuracy'}, sourceTable.Properties.VariableNames))
    error('CSV must contain channel, snr, and accuracy columns.');
end
if isempty(sourceTable) ...
        || any(ismissing(sourceTable.channel) | strlength(sourceTable.channel) == 0)
    error('Channel labels must be nonempty.');
end
validateattributes(sourceTable.snr, {'numeric'}, {'real', 'finite', 'vector'});
validateattributes(sourceTable.accuracy, {'numeric'}, ...
    {'real', 'finite', 'vector', '>=', 0, '<=', 1});
channelNames = string(option(options, 'ChannelOrder', unique(sourceTable.channel, 'stable')));
channelNames = channelNames(:);
numChannels = numel(channelNames);
if isempty(channelNames) || numel(unique(channelNames)) ~= numChannels ...
        || ~isequal(sort(channelNames), sort(unique(sourceTable.channel)))
    error('ChannelOrder must contain every measured channel exactly once.');
end
channelLabels = string(option(options, 'ChannelLabels', channelNames));
channelLabels = channelLabels(:);
if numel(channelLabels) ~= numChannels ...
        || any(ismissing(channelLabels) | strlength(channelLabels) == 0)
    error('ChannelLabels must provide one nonempty display name per channel.');
end
snr = (-20:2:18)';
accuracyFractions = zeros(numChannels, numel(snr));
for channelIdx = 1:numChannels
    rows = sourceTable(sourceTable.channel == channelNames(channelIdx), :);
    [measuredSNR, order] = sort(rows.snr);
    if ~isequal(measuredSNR(:), snr)
        error('Each channel must contain one row per -20:2:18 dB level: %s.', ...
            channelNames(channelIdx));
    end
    accuracyFractions(channelIdx, :) = rows.accuracy(order)';
end
plottedAccuracy = 100 * accuracyFractions;
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

figureSize = option(options, 'FigureSize', [960, 800]);
axesPosition = option(options, 'AxesPosition', [0.11, 0.10, 0.84, 0.84]);
fontSize = option(options, 'FontSize', 14);
axisLabelFontSize = option(options, 'AxisLabelFontSize', 14);
legendFontSize = option(options, 'LegendFontSize', 12);
legendColumns = option(options, 'LegendColumns', 2);
lineWidth = option(options, 'LineWidth', 1.1);
markerSize = option(options, 'MarkerSize', 3.5);
gridAlpha = option(options, 'GridAlpha', 1);
xLimits = option(options, 'XLimits', [-20.5, 18.5]);
yLimits = option(options, 'YLimits', [-2, 102]);
validateattributes(figureSize, {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 2});
validateattributes(axesPosition, {'numeric'}, ...
    {'real', 'finite', 'nonnegative', 'vector', 'numel', 4});
validateattributes([fontSize, axisLabelFontSize, legendFontSize, lineWidth, markerSize], ...
    {'numeric'}, {'real', 'finite', 'positive', 'vector', 'numel', 5});
validateattributes(legendColumns, {'numeric'}, {'scalar', 'integer', 'positive'});
validateattributes(gridAlpha, {'numeric'}, {'scalar', 'finite', '>=', 0, '<=', 1});
validateattributes(xLimits, {'numeric'}, {'real', 'finite', 'vector', 'numel', 2});
validateattributes(yLimits, {'numeric'}, {'real', 'finite', 'vector', 'numel', 2});
if xLimits(1) >= xLimits(2) || yLimits(1) >= yLimits(2) ...
        || xLimits(1) > min(snr) || xLimits(2) < max(snr) ...
        || yLimits(1) > min(plottedAccuracy, [], 'all') ...
        || yLimits(2) < max(plottedAccuracy, [], 'all')
    error('XLimits/YLimits must increase and include every measured point.');
end
colors = option(options, 'Colors', amr_class_colors(numChannels));
validateattributes(colors, {'numeric'}, ...
    {'real', 'finite', 'size', [numChannels, 3], '>=', 0, '<=', 1});
markers = {'o', 's', '^', 'd', 'v', '>', '<', 'p', 'h'};

fig = figure('Color', 'white', 'Units', 'pixels', ...
    'Position', [100, 80, figureSize(:)'], 'Renderer', 'painters');
ax = axes(fig, 'Units', 'normalized', 'Position', axesPosition, ...
    'Tag', 'channel_snr_axes');
hold(ax, 'on');
curves = gobjects(numChannels, 1);
for channelIdx = 1:numChannels
    curves(channelIdx) = plot(ax, snr, plottedAccuracy(channelIdx, :), ...
        'Color', colors(channelIdx, :), ...
        'Marker', markers{mod(channelIdx - 1, numel(markers)) + 1}, ...
        'MarkerFaceColor', colors(channelIdx, :), ...
        'MarkerEdgeColor', colors(channelIdx, :), ...
        'MarkerIndices', 1:numel(snr), 'LineWidth', lineWidth, ...
        'MarkerSize', markerSize, 'DisplayName', channelLabels(channelIdx), ...
        'Tag', 'channel_snr_curve');
end
set(ax, 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'GridLineStyle', ':', 'GridAlpha', gridAlpha, 'GridColor', 'black', ...
    'FontName', 'Times New Roman', 'FontSize', fontSize, ...
    'XColor', 'black', 'YColor', 'black', 'TickDir', 'in', ...
    'LineWidth', 0.8, 'Layer', 'top', 'TickLabelInterpreter', 'none');
xlim(ax, xLimits);
ylim(ax, yLimits);
xticks(ax, -20:2:18);
yticks(ax, 0:20:100);
xlabel(ax, 'SNR (dB)', 'FontName', 'Times New Roman', ...
    'FontSize', axisLabelFontSize, 'Color', 'black');
ylabel(ax, 'Accuracy (%)', 'FontName', 'Times New Roman', ...
    'FontSize', axisLabelFontSize, 'Color', 'black');
lgd = legend(ax, curves, cellstr(channelLabels), ...
    'Location', option(options, 'LegendLocation', 'southeast'), ...
    'NumColumns', min(legendColumns, numChannels), 'FontName', 'Times New Roman', ...
    'FontSize', legendFontSize, 'TextColor', 'black', 'Box', 'on', ...
    'Interpreter', 'none', 'AutoUpdate', 'off');
lgd.ItemTokenSize = option(options, 'LegendItemTokenSize', [16, 10]);
if isfield(options, 'LegendPosition') && ~isempty(options.LegendPosition)
    lgd.Units = 'normalized';
    lgd.Position = options.LegendPosition;
end
ax.Position = axesPosition;

curveData = struct('channelNames', channelNames, 'channelLabels', channelLabels, ...
    'snr', snr, 'accuracyFractions', accuracyFractions, ...
    'plottedAccuracy', plottedAccuracy, 'accuracyScale', 'percent', ...
    'sourceFile', csvFile, 'sourceTable', sourceTable, ...
    'provenanceFile', provenanceFile, 'provenance', provenance, ...
    'colors', colors, 'markers', {markers}, 'lineWidth', lineWidth, ...
    'markerSize', markerSize, 'xLimits', xLimits, 'yLimits', yLimits, ...
    'smoothingApplied', false);
fig.UserData = curveData;
drawnow;
[outDir, baseName] = fileparts(outputPath);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end
exportgraphics(fig, outputPath, 'Resolution', 300, 'BackgroundColor', 'white');
exportgraphics(fig, fullfile(outDir, [baseName, '.pdf']), ...
    'ContentType', 'vector', 'BackgroundColor', 'white');
savefig(fig, fullfile(outDir, [baseName, '.fig']));
fprintf('Saved per-channel SNR curves: %s (.png, .pdf, .fig).\n', fullfile(outDir, baseName));
end

function value = option(options, name, default)
if isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default;
end
end
