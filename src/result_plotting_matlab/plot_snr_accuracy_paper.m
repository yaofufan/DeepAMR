function plot_snr_accuracy_paper(experiments, outputPath)
%PLOT_SNR_ACCURACY_PAPER Plot measured SNR accuracy curves without smoothing.
%
% The original two-argument interface accepts CSV or metrics JSON files.
% Optional global settings are read from experiments(1):
%   figureSize, accuracyPercent, legendLocation, legendNumColumns,
%   showPanelLabel, exportFormats, strictSNRGrid, squareAxes, markers, xPadding.
% Each panel may provide colors, highlightModelIndex, insetXLim and insetYLim.
% filledMarkers, tickDirection and connectInset configure curve/zoom styling.
% fontSize, axisLabelFontSize, detailFontSize and legendFontSize control text.
% textColor defaults to pure black, [0 0 0], matching the Slides body text.
% Every measured SNR point has a marker in both the main plot and the inset.
% Accuracy data and insetYLim always use fractions in [0,1], including when
% accuracyPercent=true. The PNG is exported at 300 DPI; exportFormats={'pdf',
% 'fig'} also writes a vector PDF and an editable MATLAB figure.

if nargin < 1 || isempty(experiments)
    experiments = default_experiment();
end
if nargin < 2 || isempty(outputPath)
    outputPath = fullfile('figures', 'snr_accuracy_paper.png');
end
if ~isstruct(experiments)
    error('experiments must be a struct array.');
end
outputPath = char(outputPath);
globalCfg = experiments(1);
numPanels = numel(experiments);
figureSize = option(globalCfg, 'figureSize', [max(560, 520 * numPanels), 560]);
validateattributes(figureSize, {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 2}, mfilename, 'figureSize');
accuracyPercent = logical_option(globalCfg, 'accuracyPercent', false);
strictSNRGrid = logical_option(globalCfg, 'strictSNRGrid', false);
showPanelLabel = logical_option(globalCfg, 'showPanelLabel', true);
squareAxes = logical_option(globalCfg, 'squareAxes', false);
tickDirection = char(option(globalCfg, 'tickDirection', 'out'));
xPadding = option(globalCfg, 'xPadding', 0);
validateattributes(xPadding, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'});
legendLocation = char(option(globalCfg, 'legendLocation', 'northwest'));
legendNumColumns = option(globalCfg, 'legendNumColumns', 1);
validateattributes(legendNumColumns, {'numeric'}, ...
    {'scalar', 'integer', 'positive'}, mfilename, 'legendNumColumns');
exportFormats = as_cell(option(globalCfg, 'exportFormats', {}));
fontSize = option(globalCfg, 'fontSize', 11);
axisLabelFontSize = option(globalCfg, 'axisLabelFontSize', 12);
detailFontSize = option(globalCfg, 'detailFontSize', 9);
legendFontSize = option(globalCfg, 'legendFontSize', 10);
textColor = option(globalCfg, 'textColor', [0, 0, 0]);
validateattributes(textColor, {'numeric'}, ...
    {'real', 'finite', 'size', [1, 3], '>=', 0, '<=', 1});
validateattributes([fontSize, axisLabelFontSize, detailFontSize, legendFontSize], ...
    {'numeric'}, {'real', 'finite', 'positive', 'vector', 'numel', 4});
scale = 1 + 99 * accuracyPercent;

% Read and validate all measured values before creating the figure.
panels = cell(numPanels, 1);
for panelIdx = 1:numPanels
    expCfg = experiments(panelIdx);
    if ~isfield(expCfg, 'metricsFiles') || ~isfield(expCfg, 'modelNames')
        error('Each experiment requires metricsFiles and modelNames.');
    end
    files = as_cell(expCfg.metricsFiles);
    names = as_cell(expCfg.modelNames);
    if isempty(files) || numel(names) ~= numel(files)
        error('modelNames and metricsFiles must have the same nonzero length.');
    end
    numModels = numel(files);
    snrCell = cell(numModels, 1);
    accCell = cell(numModels, 1);
    for modelIdx = 1:numModels
        [snrCell{modelIdx}, accCell{modelIdx}] = read_snr_accuracy(files{modelIdx});
        if strictSNRGrid && modelIdx > 1 ...
                && ~isequal(snrCell{modelIdx}, snrCell{1})
            error('All model curves must share the same SNR grid: %s', files{modelIdx});
        end
    end
    if strictSNRGrid && panelIdx > 1 ...
            && ~isequal(snrCell{1}, panels{1}.snr{1})
        error('All panels must share the same SNR grid.');
    end

    highlight = option(expCfg, 'highlightModelIndex', 0);
    validateattributes(highlight, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative', '<=', numModels}, ...
        mfilename, 'highlightModelIndex');
    colors = option(expCfg, 'colors', default_colors(numModels));
    validateattributes(colors, {'numeric'}, ...
        {'real', 'finite', 'size', [numModels, 3], '>=', 0, '<=', 1}, ...
        mfilename, 'colors');
    if highlight > 0 && (~isfield(expCfg, 'colors') || isempty(expCfg.colors))
        colors(highlight, :) = [0.82, 0.12, 0.12];
    end
    insetXLim = option(expCfg, 'insetXLim', []);
    insetYLim = option(expCfg, 'insetYLim', []);
    if xor(isempty(insetXLim), isempty(insetYLim))
        error('Provide both insetXLim and insetYLim, or neither.');
    end
    if ~isempty(insetXLim)
        validate_limits(insetXLim, 'insetXLim');
        validate_limits(insetYLim, 'insetYLim');
        if any(insetYLim < 0 | insetYLim > 1)
            error('insetYLim must use accuracy fractions in [0,1].');
        end
    end
    panels{panelIdx} = struct('cfg', expCfg, 'names', {names}, ...
        'snr', {snrCell}, 'accuracy', {accCell}, 'colors', colors, ...
        'highlight', highlight, 'insetXLim', insetXLim, 'insetYLim', insetYLim);
end

outDir = fileparts(outputPath);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end
fig = figure('Color', 'w', 'Units', 'pixels', ...
    'Position', [80, 80, figureSize(:)']);
if numPanels > 1
    layout = tiledlayout(fig, 1, numPanels, ...
        'TileSpacing', 'compact', 'Padding', 'compact');
end
mainAxes = gobjects(numPanels, 1);
markers = as_cell(option(globalCfg, 'markers', ...
    {'o', 's', '^', 'd', 'v', 'p', 'h', 'x', '+', '>', '<', '*'}));
styles = {'-', '--', ':', '-.'};

for panelIdx = 1:numPanels
    data = panels{panelIdx};
    if numPanels == 1
        ax = axes(fig, 'Units', 'normalized', 'Position', [0.11, 0.10, 0.84, 0.84]);
        if squareAxes
            pbaspect(ax, [1, 1, 1]);
        end
    else
        ax = nexttile(layout);
    end
    mainAxes(panelIdx) = ax;
    style_axes(ax, fontSize, tickDirection, textColor);
    curves = gobjects(numel(data.names), 1);
    for modelIdx = 1:numel(data.names)
        curves(modelIdx) = draw_curve(ax, data, modelIdx, scale, markers, styles, false);
    end
    xlabel(ax, 'SNR (dB)', 'FontName', 'Times New Roman', ...
        'FontSize', axisLabelFontSize, 'Color', textColor);
    if accuracyPercent
        ylabel(ax, 'Accuracy (%)', 'FontName', 'Times New Roman', ...
            'FontSize', axisLabelFontSize, 'Color', textColor);
    else
        ylabel(ax, 'Accuracy', 'FontName', 'Times New Roman', ...
            'FontSize', axisLabelFontSize, 'Color', textColor);
    end
    textLabel = char(option(data.cfg, 'title', ''));
    if showPanelLabel
        textLabel = subplot_title(panelIdx, textLabel);
    end
    if ~isempty(textLabel)
        title(ax, textLabel, 'FontName', 'Times New Roman', ...
            'FontSize', axisLabelFontSize, 'FontWeight', 'normal', ...
            'Color', textColor, 'Interpreter', 'none');
    end
    xlim(ax, [-20 - xPadding, 18 + xPadding]);
    xticks(ax, -20:2:18);
    ylim(ax, [0, scale]);
    yticks(ax, (0:0.2:1) * scale);
    lgd = legend(ax, curves, data.names, 'Location', legendLocation, ...
        'NumColumns', legendNumColumns, 'FontName', 'Times New Roman', ...
        'FontSize', legendFontSize, 'Box', 'on', 'Interpreter', 'none', 'AutoUpdate', 'off');
    if isfield(globalCfg, 'legendItemTokenSize') && ~isempty(globalCfg.legendItemTokenSize)
        lgd.ItemTokenSize = globalCfg.legendItemTokenSize;
    end
    lgd.TextColor = textColor;
    lgd.EdgeColor = textColor;
end

% Outside legends change tile geometry. Position insets only after layout.
drawnow;
for panelIdx = 1:numPanels
    data = panels{panelIdx};
    if ~isempty(data.insetXLim)
        draw_inset(fig, mainAxes(panelIdx), data, scale, markers, styles, detailFontSize, textColor);
    end
end
drawnow;
exportgraphics(fig, outputPath, 'Resolution', 300);
fprintf('Saved paper-style SNR figure to %s\n', outputPath);
[baseDir, baseName] = fileparts(outputPath);
basePath = fullfile(baseDir, baseName);
for formatIdx = 1:numel(exportFormats)
    format = lower(exportFormats{formatIdx});
    switch format
        case 'pdf'
            path = [basePath, '.pdf'];
            exportgraphics(fig, path, 'ContentType', 'vector');
        case 'fig'
            path = [basePath, '.fig'];
            savefig(fig, path);
        case 'png'
            path = [basePath, '.png'];
            if ~strcmp(path, outputPath)
                exportgraphics(fig, path, 'Resolution', 300);
            end
        otherwise
            error('Unsupported export format: %s. Use pdf, fig or png.', format);
    end
    fprintf('Saved %s\n', path);
end
end

function expCfg = default_experiment()
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fullfile(scriptDir, '..', '..');
expCfg.title = 'MATLAB AMR Dataset';
expCfg.metricsFiles = {fullfile(projectRoot, 'src', 'runs', ...
    'eval_oshea_resnet', 'snr_accuracy.csv')};
expCfg.modelNames = {'ResNet'};
expCfg.insetXLim = [0, 18];
expCfg.insetYLim = [0.65, 0.90];
end

function value = option(cfg, name, fallback)
if isfield(cfg, name) && ~isempty(cfg.(name))
    value = cfg.(name);
else
    value = fallback;
end
end

function value = logical_option(cfg, name, fallback)
value = option(cfg, name, fallback);
if ~(isscalar(value) && (islogical(value) ...
        || (isnumeric(value) && isfinite(value) && any(value == [0, 1]))))
    error('%s must be a logical scalar.', name);
end
value = logical(value);
end

function value = as_cell(value)
if ischar(value) || isstring(value)
    value = cellstr(value);
end
if ~iscell(value) || ~all(cellfun(@(x) ischar(x) ...
        || (isstring(x) && isscalar(x)), value))
    error('File names, model names and export formats must be text lists.');
end
value = cellfun(@char, value(:)', 'UniformOutput', false);
end

function colors = default_colors(numModels)
% Distinct categorical colors, including all eleven models without an
% eight-color cycle. An explicit experiments.colors palette takes priority.
palette = [
    0.12, 0.47, 0.71;
    1.00, 0.50, 0.05;
    0.17, 0.63, 0.17;
    0.58, 0.40, 0.74;
    0.55, 0.34, 0.29;
    0.89, 0.47, 0.76;
    0.45, 0.45, 0.45;
    0.74, 0.74, 0.13;
    0.09, 0.65, 0.75;
    0.15, 0.15, 0.15;
    0.82, 0.12, 0.12;
    0.35, 0.62, 0.46
];
if numModels > size(palette, 1)
    palette = [palette; hsv(numModels - size(palette, 1))];
end
colors = palette(1:numModels, :);
end

function label = subplot_title(panelIdx, textLabel)
letters = 'abcdefghijklmnopqrstuvwxyz';
if panelIdx <= numel(letters)
    label = sprintf('(%s) %s', letters(panelIdx), textLabel);
else
    label = sprintf('(%d) %s', panelIdx, textLabel);
end
end

function validate_limits(limits, name)
validateattributes(limits, {'numeric'}, ...
    {'real', 'finite', 'vector', 'numel', 2}, mfilename, name);
if limits(1) >= limits(2)
    error('%s must contain increasing limits.', name);
end
end

function [snrValues, accuracy] = read_snr_accuracy(metricsFile)
metricsFile = char(metricsFile);
if ~exist(metricsFile, 'file')
    error('Metrics file not found: %s', metricsFile);
end
[~, ~, ext] = fileparts(metricsFile);
if strcmpi(ext, '.csv')
    tableData = readtable(metricsFile);
    if ~all(ismember({'snr', 'accuracy'}, tableData.Properties.VariableNames))
        error('CSV requires snr and accuracy columns: %s', metricsFile);
    end
    snrValues = tableData.snr;
    accuracy = tableData.accuracy;
else
    text = fileread(metricsFile);
    bodyMatch = regexp(text, '"per_snr_accuracy"\s*:\s*\{([\s\S]*?)\}', 'tokens', 'once');
    if isempty(bodyMatch)
        error('Cannot find per_snr_accuracy in %s', metricsFile);
    end
    pairMatches = regexp(bodyMatch{1}, '"(?<snr>-?\d+)"\s*:\s*(?<acc>[0-9.eE+-]+)', 'names');
    if isempty(pairMatches)
        error('Cannot parse per_snr_accuracy values in %s', metricsFile);
    end
    snrValues = arrayfun(@(p) str2double(p.snr), pairMatches(:));
    accuracy = arrayfun(@(p) str2double(p.acc), pairMatches(:));
end
if ~isnumeric(snrValues) || ~isnumeric(accuracy) ...
        || isempty(snrValues) || numel(snrValues) ~= numel(accuracy)
    error('SNR and accuracy must be nonempty numeric vectors of equal length: %s', metricsFile);
end
snrValues = snrValues(:);
accuracy = accuracy(:);
if ~isreal(snrValues) || ~isreal(accuracy) ...
        || any(~isfinite(snrValues)) || any(~isfinite(accuracy)) ...
        || any(accuracy < 0 | accuracy > 1)
    error('SNR must be finite and accuracy must be finite fractions in [0,1]: %s', metricsFile);
end
if numel(unique(snrValues)) ~= numel(snrValues)
    error('SNR values must be unique: %s', metricsFile);
end
[snrValues, order] = sort(snrValues);
accuracy = accuracy(order);
end

function style_axes(ax, fontSize, tickDirection, textColor)
hold(ax, 'on');
box(ax, 'on');
grid(ax, 'on');
ax.GridLineStyle = ':';
ax.GridColor = textColor;
ax.GridAlpha = 1;
ax.LineWidth = 0.8;
ax.FontName = 'Times New Roman';
ax.FontSize = fontSize;
ax.TickDir = tickDirection;
ax.XColor = textColor;
ax.YColor = textColor;
end

function curve = draw_curve(ax, data, modelIdx, scale, markers, styles, isInset)
isHighlight = modelIdx == data.highlight;
lineWidth = 1.3;
markerSize = 4.2;
if isHighlight
    markerSize = 5.0;
end
if isInset
    lineWidth = 1.0;
    markerSize = 2.8 + 0.6 * isHighlight;
end
numPoints = numel(data.snr{modelIdx});
markerIndices = 1:numPoints;
style = styles{mod(modelIdx - 1, numel(styles)) + 1};
if isHighlight
    style = '-';
end
color = data.colors(modelIdx, :);
curve = plot(ax, data.snr{modelIdx}, data.accuracy{modelIdx} * scale, ...
    'Color', color, 'LineStyle', style, ...
    'Marker', markers{mod(modelIdx - 1, numel(markers)) + 1}, ...
    'MarkerIndices', markerIndices, 'LineWidth', lineWidth, ...
    'MarkerSize', markerSize, 'DisplayName', data.names{modelIdx});
if isHighlight || logical_option(data.cfg, 'filledMarkers', false)
    curve.MarkerFaceColor = color;
else
    curve.MarkerFaceColor = 'none';
end
end

function draw_inset(fig, parentAx, data, scale, markers, styles, detailFontSize, textColor)
parentPos = parentAx.Position;
insetPos = [
    parentPos(1) + parentPos(3) * 0.51, ...
    parentPos(2) + parentPos(4) * 0.10, ...
    parentPos(3) * 0.44, ...
    parentPos(4) * 0.38
];
if logical_option(data.cfg, 'connectInset', false)
    % Outline the measured zoom region in data coordinates. Link its lower
    % corners to the inset's upper corners in figure coordinates.
    zoomYLim = data.insetYLim * scale;
    rectangle('Parent', parentAx, 'Position', ...
        [data.insetXLim(1), zoomYLim(1), diff(data.insetXLim), diff(zoomYLim)], ...
        'LineStyle', '--', 'LineWidth', 0.9, 'EdgeColor', textColor, ...
        'HandleVisibility', 'off', 'Tag', 'snr_zoom_region');
    sourceX = parentPos(1) + parentPos(3) * ...
        (data.insetXLim - parentAx.XLim(1)) / diff(parentAx.XLim);
    sourceY = parentPos(2) + parentPos(4) * ...
        (zoomYLim(1) - parentAx.YLim(1)) / diff(parentAx.YLim);
    targetX = [insetPos(1), insetPos(1) + insetPos(3)];
    targetY = insetPos(2) + insetPos(4);
    for cornerIdx = 1:2
        annotation(fig, 'line', [sourceX(cornerIdx), targetX(cornerIdx)], ...
            [sourceY, targetY], 'LineStyle', '--', 'LineWidth', 0.9, ...
            'Color', textColor, 'Tag', 'snr_inset_connector');
    end
end
insetAx = axes(fig, 'Units', 'normalized', 'Position', insetPos, 'Color', 'w');
style_axes(insetAx, detailFontSize, ...
    char(option(data.cfg, 'tickDirection', 'out')), textColor);
for modelIdx = 1:numel(data.names)
    draw_curve(insetAx, data, modelIdx, scale, markers, styles, true);
end
xlim(insetAx, data.insetXLim);
if isequal(data.insetXLim(:)', [0, 18])
    xticks(insetAx, 0:2:18);
else
    xticks(insetAx, linspace(data.insetXLim(1), data.insetXLim(2), 4));
end
ylim(insetAx, data.insetYLim * scale);
if scale == 100
    lowerTick = ceil(data.insetYLim(1) * 100 / 2) * 2;
    upperTick = floor(data.insetYLim(2) * 100 / 2) * 2;
    yticks(insetAx, lowerTick:2:upperTick);
end
end
