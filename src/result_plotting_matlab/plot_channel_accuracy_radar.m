function [fig, comparison] = plot_channel_accuracy_radar(experiment, outputPath)
%PLOT_CHANNEL_ACCURACY_RADAR Compare every model across channel conditions.
% Required fields: metricsFiles and modelNames. Each input is an evaluation
% channel_accuracy.csv with accuracy fractions in [0,1]. Rows are aligned by
% channel name. Curves contain all channels and close at their first point.
% Optional fields: channelOrder, channelLabels, figureSize, colors, markers,
% lineStyles, lineWidth, markerSize, axisLimits, exportFormats.
% fontSize, detailFontSize and legendFontSize control labels and legend text.
% axesPosition, xLimits, yLimits, rangeLabelOffset, bottomLabelRadius and
% legendBottom control layout within the fixed canvas.
% channelLabels may contain line breaks; channelLabelOffsets is an optional
% numChannels-by-2 matrix of label displacements in the radar's data units.
% Each spoke has independent bounds, printed below its channel label in %.
% Default bounds extend one observed range below the minimum and round
% outward to 0.1 percentage points. The best value lies near the outer ring.
% Optional axisLimits is a numChannels-by-2 matrix in percentage units.
% textColor defaults to pure black, [0 0 0], matching the Slides body text.
% spokeLineStyle controls the radial guides (default '--').
% spokeOpacity softens only the radial guides (default 1.0). It is composited
% against the white background as RGB to preserve native vector PDF output.
% Outputs: 300-DPI PNG, vector PDF and editable FIG by default. The returned
% table and FIG UserData retain all unrounded per-channel accuracy values.

if nargin < 2 || isempty(outputPath)
    outputPath = fullfile('figures', 'channel_accuracy_radar.png');
end
required = {'metricsFiles', 'modelNames'};
if ~isstruct(experiment) || ~isscalar(experiment) ...
        || ~all(isfield(experiment, required))
    error('Provide one experiment with metricsFiles and modelNames.');
end
files = string(experiment.metricsFiles(:));
names = string(experiment.modelNames(:));
if isempty(files) || numel(files) ~= numel(names) ...
        || any(ismissing(names) | strlength(names) == 0)
    error('metricsFiles and modelNames must have the same nonzero length.');
end
if numel(unique(names)) ~= numel(names)
    error('Model display names must be unique.');
end
first = read_channel_accuracy(files(1));
channels = string(option(experiment, 'channelOrder', first.channel));
channels = channels(:);
numChannels = numel(channels);
numModels = numel(files);
if numChannels < 3 || numChannels ~= height(first) ...
        || numel(unique(channels)) ~= numChannels ...
        || ~all(ismember(channels, first.channel))
    error('channelOrder must contain every input channel exactly once (at least three).');
end
labels = string(option(experiment, 'channelLabels', channels));
labels = labels(:);
if numel(labels) ~= numChannels
    error('channelLabels must match channelOrder in length.');
end
% Preserve multiline channel names; numeric ranges use separate text objects.
accuracy = zeros(numChannels, numModels);
for modelIdx = 1:numModels
    data = read_channel_accuracy(files(modelIdx));
    [matched, order] = ismember(channels, data.channel);
    if height(data) ~= numChannels || ~all(matched)
        error('All models must contain the same channel set: %s', files(modelIdx));
    end
    accuracy(:, modelIdx) = 100 * data.accuracy(order);
end
channelMin = min(accuracy, [], 2);
channelMax = max(accuracy, [], 2);
spread = max(channelMax - channelMin, 0.1);
lowerBounds = max(0, floor((channelMin - spread) * 10) / 10);
upperBounds = min(100, ceil(channelMax * 10) / 10);
upperBounds = max(upperBounds, lowerBounds + 0.1);
axisLimits = option(experiment, 'axisLimits', [lowerBounds, upperBounds]);
validateattributes(axisLimits, {'numeric'}, ...
    {'real', 'finite', 'size', [numChannels, 2], '>=', 0, '<=', 100});
if any(axisLimits(:, 2) <= axisLimits(:, 1))
    error('Every axis upper bound must exceed its lower bound.');
end
if any(accuracy < axisLimits(:, 1) | accuracy > axisLimits(:, 2), 'all')
    error('axisLimits would clip measured accuracy values. Expand the bounds.');
end
radius = (accuracy - axisLimits(:, 1)) ./ (axisLimits(:, 2) - axisLimits(:, 1));
figureSize = option(experiment, 'figureSize', [960, 800]);
validateattributes(figureSize, {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 2});
colors = option(experiment, 'colors', lines(numModels));
validateattributes(colors, {'numeric'}, ...
    {'real', 'finite', 'size', [numModels, 3], '>=', 0, '<=', 1});
markers = string(option(experiment, 'markers', {'o', 's', '^', 'd', 'v', 'p', 'h', '>', '<'}));
styles = string(option(experiment, 'lineStyles', {'-', '--', ':', '-.'}));
lineWidth = option(experiment, 'lineWidth', 1.1);
markerSize = option(experiment, 'markerSize', 3.5);
fontSize = option(experiment, 'fontSize', 11);
detailFontSize = option(experiment, 'detailFontSize', 9.5);
legendFontSize = option(experiment, 'legendFontSize', 10);
textColor = option(experiment, 'textColor', [0, 0, 0]);
spokeOpacity = option(experiment, 'spokeOpacity', 1.0);
spokeLineStyle = validatestring(option(experiment, 'spokeLineStyle', '--'), ...
    {'-', '--', ':', '-.', 'none'});
validateattributes(spokeOpacity, {'numeric'}, ...
    {'real', 'finite', 'scalar', '>=', 0, '<=', 1});
spokeColor = (1 - spokeOpacity) * [1, 1, 1] + spokeOpacity * textColor;
validateattributes(textColor, {'numeric'}, ...
    {'real', 'finite', 'size', [1, 3], '>=', 0, '<=', 1});
validateattributes(lineWidth, {'numeric'}, {'scalar', 'positive', 'finite'});
validateattributes(markerSize, {'numeric'}, {'scalar', 'positive', 'finite'});
validateattributes([fontSize, detailFontSize, legendFontSize], ...
    {'numeric'}, {'real', 'finite', 'positive', 'vector', 'numel', 3});
axesPosition = option(experiment, 'axesPosition', [0.065, 0.18, 0.87, 0.80]);
xLimits = option(experiment, 'xLimits', [-1.48, 1.48]);
yLimits = option(experiment, 'yLimits', [-1.20, 1.26]);
legendBottom = option(experiment, 'legendBottom', 0.09);
rangeLabelOffset = option(experiment, 'rangeLabelOffset', -0.012);
bottomLabelRadius = option(experiment, 'bottomLabelRadius', 1.16);
channelLabelOffsets = option(experiment, 'channelLabelOffsets', zeros(numChannels, 2));
validateattributes(channelLabelOffsets, {'numeric'}, ...
    {'real', 'finite', 'size', [numChannels, 2]});

fig = figure('Color', 'w', 'Units', 'pixels', ...
    'Position', [80, 60, figureSize(:)']);
% Cartesian primitives support separate two-line labels on every spoke and
% keep the outer circle, rays and all curves as vector objects in the PDF.
ax = axes(fig, 'Units', 'normalized', 'Position', axesPosition, ...
    'XLim', xLimits, 'YLim', yLimits, 'DataAspectRatio', [1, 1, 1], ...
    'Visible', 'off', 'Tag', 'channel_radar_axes');
hold(ax, 'on');
theta = pi / 2 - (0:numChannels - 1)' * 2 * pi / numChannels;
circleTheta = linspace(0, 2 * pi, 361);
plot(ax, cos(circleTheta), sin(circleTheta), ...
    'Color', textColor, 'LineWidth', 0.8, ...
    'HandleVisibility', 'off', 'Tag', 'channel_radar_outer_circle');
for channelIdx = 1:numChannels
    line(ax, [0, cos(theta(channelIdx))], [0, sin(theta(channelIdx))], ...
        'Color', spokeColor, 'LineStyle', spokeLineStyle, ...
        'LineWidth', 0.7, 'HandleVisibility', 'off', ...
        'Tag', 'channel_radar_spoke');
    anchorX = 1.08 * cos(theta(channelIdx));
    anchorY = 1.08 * sin(theta(channelIdx));
    if sin(theta(channelIdx)) < -0.90
        anchorY = bottomLabelRadius * sin(theta(channelIdx));
    end
    anchorX = anchorX + channelLabelOffsets(channelIdx, 1);
    anchorY = anchorY + channelLabelOffsets(channelIdx, 2);
    if cos(theta(channelIdx)) > 0.35
        alignment = 'left';
    elseif cos(theta(channelIdx)) < -0.35
        alignment = 'right';
    else
        alignment = 'center';
    end
    text(ax, anchorX, anchorY + 0.020, labels(channelIdx), ...
        'HorizontalAlignment', alignment, 'VerticalAlignment', 'bottom', ...
        'FontName', 'Times New Roman', 'FontSize', fontSize, ...
        'Color', textColor, ...
        'Interpreter', 'none', 'Clipping', 'off', ...
        'Tag', 'channel_radar_channel_label');
    rangeLabel = sprintf('(%.1f%%, %.1f%%)', axisLimits(channelIdx, :));
    text(ax, anchorX, anchorY + rangeLabelOffset, rangeLabel, ...
        'HorizontalAlignment', alignment, 'VerticalAlignment', 'top', ...
        'FontName', 'Times New Roman', 'FontSize', detailFontSize, ...
        'Color', textColor, ...
        'Interpreter', 'none', 'Clipping', 'off', ...
        'Tag', 'channel_radar_range_label');
end
curves = gobjects(numModels, 1);
for modelIdx = 1:numModels
    values = radius(:, modelIdx);
    x = values .* cos(theta);
    y = values .* sin(theta);
    curves(modelIdx) = plot(ax, [x; x(1)], [y; y(1)], ...
        'Color', colors(modelIdx, :), ...
        'LineStyle', styles(mod(modelIdx - 1, numel(styles)) + 1), ...
        'LineWidth', lineWidth, ...
        'Marker', markers(mod(modelIdx - 1, numel(markers)) + 1), ...
        'MarkerSize', markerSize, 'MarkerIndices', 1:numChannels, ...
        'MarkerFaceColor', colors(modelIdx, :), ...
        'MarkerEdgeColor', colors(modelIdx, :), ...
        'DisplayName', names(modelIdx), 'Tag', 'channel_radar_curve');
end

lgd = legend(ax, curves, cellstr(names), 'NumColumns', 5, 'Location', 'southoutside', ...
    'FontName', 'Times New Roman', 'FontSize', legendFontSize, 'Box', 'on', ...
    'Interpreter', 'none', 'AutoUpdate', 'off');
lgd.Units = 'normalized';
lgd.ItemTokenSize = option(experiment, 'legendItemTokenSize', [24, 10]);
lgd.LineWidth = 0.7;
lgd.Color = 'white';
lgd.TextColor = textColor;
lgd.EdgeColor = textColor;
drawnow;
legendPosition = lgd.Position;
legendPosition(1) = (1 - legendPosition(3)) / 2;
legendPosition(2) = legendBottom;
lgd.Position = legendPosition;
ax.Position = axesPosition;

comparison = array2table(accuracy, ...
    'VariableNames', cellstr(matlab.lang.makeValidName(names)));
comparison = addvars(comparison, channels, 'Before', 1, 'NewVariableNames', 'Channel');
fig.UserData = struct('channelNames', channels, 'modelNames', names, ...
    'accuracyPercent', accuracy, 'metricsFiles', files, 'comparison', comparison, ...
    'axisLimitsPercent', axisLimits, 'normalizedRadius', radius);
[outputDir, baseName] = fileparts(char(outputPath));
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
drawnow;
exportFormats = string(option(experiment, 'exportFormats', {'pdf', 'fig'}));
exportgraphics(fig, outputPath, 'Resolution', 300);
fprintf('Saved channel radar figure to %s\n', outputPath);
basePath = fullfile(outputDir, baseName);
for format = exportFormats(:)'
    path = [basePath, '.', char(lower(format))];
    switch lower(format)
        case "pdf"
            exportgraphics(fig, path, 'ContentType', 'vector');
        case "fig"
            savefig(fig, path);
        case "png"
            if ~strcmp(path, outputPath)
                exportgraphics(fig, path, 'Resolution', 300);
            end
        otherwise
            error('Unsupported export format: %s. Use pdf, fig or png.', format);
    end
    fprintf('Saved %s\n', path);
end
end

function data = read_channel_accuracy(path)
if ~isfile(path)
    error('Result file not found: %s', path);
end
data = readtable(path, 'TextType', 'string');
if ~all(ismember({'channel', 'accuracy'}, data.Properties.VariableNames))
    error('CSV must contain channel and accuracy columns: %s', path);
end
if isempty(data) || any(ismissing(data.channel) | strlength(data.channel) == 0) ...
        || numel(unique(data.channel)) ~= height(data)
    error('Channel names must be nonempty and unique: %s', path);
end
validateattributes(data.accuracy, {'numeric'}, ...
    {'real', 'finite', 'vector', '>=', 0, '<=', 1}, mfilename, 'accuracy');
end

function value = option(cfg, name, default)
if isfield(cfg, name) && ~isempty(cfg.(name))
    value = cfg.(name);
else
    value = default;
end
end
