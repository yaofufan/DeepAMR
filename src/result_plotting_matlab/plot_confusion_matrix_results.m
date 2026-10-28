function [fig, matrixData] = plot_confusion_matrix_results(csvFile, outputPath, options)
%PLOT_CONFUSION_MATRIX_RESULTS Plot counts or row-normalized percentages.
% The default preserves the original count plot and PNG-only export.
% Normalization: 'none' (counts) or 'row' (percentage within each true class).
% AnnotationThreshold and ShowZeros affect text only, never the plotted data.
% Title may be empty. FigureSize, AxesPosition, FontSize, CellFontSize and
% AxisLabelFontSize control layout. ExportFormats accepts 'pdf' and 'fig'.
% VectorCells=true also renders count matrices as RGB rectangles. Row mode
% always uses vector cells. ColorbarLabel may be empty; ColorbarExponent
% optionally controls scientific notation on the color scale.
% FIG UserData retains counts, all percentages, class order and source path.
% Percentage matrices and their color scale use individual RGB rectangles,
% avoiding colormap-shaded meshes and image-based colorbars in vector PDFs.

arguments
    csvFile (1, :) char
    outputPath (1, :) char
    options struct = struct()
end

if ~isfield(options, 'Title')
    options.Title = 'Confusion Matrix';
end
options = apply_default(options, 'Normalization', 'none');
options = apply_default(options, 'AnnotationThreshold', 0);
options = apply_default(options, 'ShowZeros', true);
options = apply_default(options, 'VectorCells', false);
options = apply_default(options, 'FontSize', 9);
options = apply_default(options, 'CellFontSize', 7);
options = apply_default(options, 'AxisLabelFontSize', options.FontSize + 1);
options = apply_default(options, 'ExportFormats', {});
normalization = validatestring(options.Normalization, {'none', 'row'});
options.Normalization = normalization;
validateattributes(options.AnnotationThreshold, {'numeric'}, ...
    {'real', 'finite', 'scalar', 'nonnegative'});
validateattributes(options.ShowZeros, {'logical', 'numeric'}, ...
    {'scalar', 'binary'});
validateattributes(options.VectorCells, {'logical', 'numeric'}, ...
    {'scalar', 'binary'});
useVectorCells = strcmp(normalization, 'row') || logical(options.VectorCells);
validateattributes([options.FontSize, options.CellFontSize, options.AxisLabelFontSize], ...
    {'numeric'}, {'real', 'finite', 'positive', 'vector', 'numel', 3});
if ~exist(csvFile, 'file')
    error('Result file not found: %s', csvFile);
end

raw = readcell(csvFile);
classNames = cellstr(string(raw(1, 2:end)));
rowNames = cellstr(string(raw(2:end, 1)));
counts = cell2mat(raw(2:end, 2:end));
numClasses = numel(classNames);
if numClasses == 0 || ~isequal(classNames(:), rowNames(:)) ...
        || numel(unique(classNames)) ~= numClasses
    error('Confusion-matrix row and column class names do not match.');
end
validateattributes(counts, {'numeric'}, ...
    {'real', 'finite', 'nonnegative', 'integer', 'size', [numClasses, numClasses]});
rowTotals = sum(counts, 2);
rowPercentage = zeros(size(counts));
nonemptyRows = rowTotals > 0;
rowPercentage(nonemptyRows, :) = 100 * counts(nonemptyRows, :) ./ rowTotals(nonemptyRows);
if strcmp(normalization, 'row')
    if any(~nonemptyRows)
        error('Row normalization requires at least one sample in every true class.');
    end
    values = rowPercentage;
    colorLimit = 100;
    options = apply_default(options, 'ValueFormat', '%.1f');
    colorbarLabel = 'Row percentage (%)';
else
    values = counts;
    colorLimit = max(max(counts, [], 'all'), 1);
    options = apply_default(options, 'ValueFormat', '%d');
    colorbarLabel = 'Sample count';
end
if isfield(options, 'ColorbarLabel')
    colorbarLabel = options.ColorbarLabel;
end

figSize = max(680, 45 * numClasses);
options = apply_default(options, 'FigureSize', [figSize, figSize]);
validateattributes(options.FigureSize, {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 2});
fig = figure('Color', 'w', 'Units', 'pixels', ...
    'Position', [100, 60, options.FigureSize(:)'], 'Renderer', 'painters');
ax = axes(fig, 'Units', 'normalized');
colors = blue_colormap(256);
if useVectorCells
    options = apply_default(options, 'AxesPosition', [0.17, 0.20, 0.68, 0.76]);
    hold(ax, 'on');
    for row = 1:numClasses
        for col = 1:numClasses
            colorIndex = min(size(colors, 1), ...
                max(1, fix(values(row, col) / colorLimit * size(colors, 1)) + 1));
            rectangle(ax, 'Position', [col - 0.5, row - 0.5, 1, 1], ...
                'FaceColor', colors(colorIndex, :), 'EdgeColor', 'none', ...
                'Tag', 'confusion_matrix_values', 'UserData', values(row, col));
        end
    end
    set(ax, 'YDir', 'reverse', 'XLim', [0.5, numClasses + 0.5], ...
        'YLim', [0.5, numClasses + 0.5], 'CLim', [0, colorLimit]);
else
    imageHandle = imagesc(ax, values, [0, colorLimit]);
    imageHandle.Tag = 'confusion_matrix_values';
end
axis(ax, 'square');
colormap(ax, colors);
if ~useVectorCells
    % Preserve the original count-mode image and standard colorbar behavior.
    cb = colorbar(ax);
    cb.FontName = 'Times New Roman';
    cb.FontSize = options.FontSize;
    cb.Color = [0, 0, 0];
    cb.Label.String = colorbarLabel;
    cb.Label.FontName = 'Times New Roman';
    cb.Label.FontSize = options.AxisLabelFontSize;
    cb.Label.Color = [0, 0, 0];
end
if isfield(options, 'AxesPosition') && ~isempty(options.AxesPosition)
    validateattributes(options.AxesPosition, {'numeric'}, ...
        {'real', 'finite', 'nonnegative', 'vector', 'numel', 4});
    ax.Position = options.AxesPosition;
end
ax.FontName = 'Times New Roman';
ax.FontSize = options.FontSize;
ax.XColor = [0, 0, 0];
ax.YColor = [0, 0, 0];
ax.TickDir = 'in';
ax.TickLength = [0.005, 0.005];
ax.LineWidth = 0.8;
ax.Box = 'on';
ax.Layer = 'top';
ax.Tag = 'confusion_matrix_axes';
ax.XTick = 1:numClasses;
ax.YTick = 1:numClasses;
ax.XTickLabel = classNames;
ax.YTickLabel = classNames;
xtickangle(ax, 45);
xlabel(ax, 'Predicted modulation', 'FontName', 'Times New Roman', ...
    'FontSize', options.AxisLabelFontSize, 'Color', [0, 0, 0]);
ylabel(ax, 'True modulation', 'FontName', 'Times New Roman', ...
    'FontSize', options.AxisLabelFontSize, 'Color', [0, 0, 0]);
title(ax, options.Title, 'FontName', 'Times New Roman', ...
    'FontSize', options.AxisLabelFontSize, 'FontWeight', 'normal', ...
    'Color', [0, 0, 0], 'Interpreter', 'none');
if useVectorCells
    draw_vector_colorbar(fig, ax, colors, colorLimit, colorbarLabel, options);
end

% Keep text dark on medium-blue percentage cells for projection contrast.
if strcmp(normalization, 'row')
    whiteTextThreshold = 0.75 * colorLimit;
else
    whiteTextThreshold = 0.55 * colorLimit;
end
for row = 1:numClasses
    for col = 1:numClasses
        if (~options.ShowZeros && counts(row, col) == 0) ...
                || values(row, col) < options.AnnotationThreshold
            continue;
        end
        textColor = [0, 0, 0];
        if values(row, col) > whiteTextThreshold
            textColor = [1, 1, 1];
        end
        text(ax, col, row, sprintf(options.ValueFormat, values(row, col)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontName', 'Times New Roman', ...
            'FontSize', options.CellFontSize, ...
            'Color', textColor, 'Interpreter', 'none', ...
            'Tag', 'confusion_matrix_annotation');
    end
end

[outputDir, baseName] = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
matrixData = struct('sourceCsv', csvFile, 'classNames', {classNames(:)}, ...
    'counts', counts, 'rowTotals', rowTotals, ...
    'rowPercentage', rowPercentage, 'displayValues', values, ...
    'normalization', normalization, ...
    'annotationThreshold', options.AnnotationThreshold, ...
    'showZeros', logical(options.ShowZeros));
if isfield(options, 'Provenance')
    matrixData.provenance = options.Provenance;
end
fig.UserData = matrixData;
drawnow;
exportgraphics(fig, outputPath, 'Resolution', 300);
fprintf('Saved confusion matrix to %s\n', outputPath);
basePath = fullfile(outputDir, baseName);
exportFormats = string(options.ExportFormats);
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

function colors = blue_colormap(numColors)
anchors = [
    1.00, 1.00, 1.00
    0.78, 0.88, 0.96
    0.36, 0.65, 0.82
    0.06, 0.28, 0.55
];
x = linspace(0, 1, size(anchors, 1));
xi = linspace(0, 1, numColors);
colors = interp1(x, anchors, xi, 'linear');
end

function draw_vector_colorbar(fig, mainAxes, colors, colorLimit, label, options)
% Match the height of the actual square plot box, not its enclosing axes.
figureSize = fig.Position(3:4);
axesPosition = mainAxes.Position;
axesSizePixels = axesPosition(3:4) .* figureSize;
sidePixels = min(axesSizePixels);
plotOriginPixels = axesPosition(1:2) .* figureSize ...
    + (axesSizePixels - sidePixels) / 2;
gapPixels = 0.026 * figureSize(1);
barWidthPixels = 0.019 * figureSize(1);
barPosition = [(plotOriginPixels(1) + sidePixels + gapPixels) / figureSize(1), ...
    plotOriginPixels(2) / figureSize(2), ...
    barWidthPixels / figureSize(1), sidePixels / figureSize(2)];
barAxes = axes(fig, 'Units', 'normalized', 'Position', barPosition, ...
    'XLim', [0, 1], 'YLim', [0, colorLimit], 'YDir', 'normal', ...
    'XTick', [], 'YAxisLocation', 'right', ...
    'FontName', 'Times New Roman', 'FontSize', options.FontSize, ...
    'XColor', [0, 0, 0], 'YColor', [0, 0, 0], 'Color', 'white', ...
    'Box', 'on', 'LineWidth', 0.8, 'TickDir', 'in', ...
    'Layer', 'top', 'Tag', 'confusion_matrix_colorbar_axes');
if strcmp(options.Normalization, 'row')
    barAxes.YTick = 0:20:100;
end
if isfield(options, 'ColorbarExponent') && ~isempty(options.ColorbarExponent)
    validateattributes(options.ColorbarExponent, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'integer'});
    barAxes.YAxis.Exponent = options.ColorbarExponent;
end
hold(barAxes, 'on');
step = colorLimit / size(colors, 1);
for colorIndex = 1:size(colors, 1)
    rectangle(barAxes, 'Position', [0, (colorIndex - 1) * step, 1, step], ...
        'FaceColor', colors(colorIndex, :), 'EdgeColor', 'none', ...
        'Tag', 'confusion_matrix_colorbar_values', ...
        'UserData', (colorIndex - 0.5) * step);
end
ylabel(barAxes, label, 'FontName', 'Times New Roman', ...
    'FontSize', options.AxisLabelFontSize, 'Color', [0, 0, 0]);
end

function options = apply_default(options, name, value)
if ~isfield(options, name) || isempty(options.(name))
    options.(name) = value;
end
end
