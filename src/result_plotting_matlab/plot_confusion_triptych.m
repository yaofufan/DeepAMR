function [fig, matrixData] = plot_confusion_triptych(counts, classNames, outputPath, options)
%PLOT_CONFUSION_TRIPTYCH Three count matrices with independent count scales.
% counts is numClasses-by-numClasses-by-3, with true classes along rows.
% All integer cells, including zeros, are annotated. Color limits default to
% the largest row sample count in each panel, never a percentage conversion.
% RGB rectangles keep both cells and color scales vector in PDF exports.

if nargin < 4
    options = struct();
end
classNames = string(classNames(:));
numClasses = numel(classNames);
if numClasses == 0 || numel(unique(classNames)) ~= numClasses ...
        || any(ismissing(classNames) | strlength(classNames) == 0)
    error('Provide a nonempty vector of unique class names.');
end
validateattributes(counts, {'numeric'}, ...
    {'real', 'finite', 'integer', 'nonnegative', 'size', [numClasses, numClasses, 3]});
options = apply_default(options, 'FigureSize', [1880, 720]);
options = apply_default(options, 'FontSize', 13);
options = apply_default(options, 'CellFontSize', 9.5);
options = apply_default(options, 'AxisLabelFontSize', 14);
options = apply_default(options, 'TitleFontSize', 20);
options = apply_default(options, 'PanelNames', {'Overall', '0 dB', '10 dB'});
options = apply_default(options, 'ExportFormats', {'pdf', 'fig'});
validateattributes(options.FigureSize, {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 2});
validateattributes([options.FontSize, options.CellFontSize, ...
    options.AxisLabelFontSize, options.TitleFontSize], {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 4});
panelNames = string(options.PanelNames(:));
assert(numel(panelNames) == 3, 'Exactly three panel names are required.');
rowTotals = squeeze(sum(counts, 2));
limits = [zeros(3, 1), max(max(rowTotals, [], 1)', 1)];
sampleCounts = reshape(sum(sum(counts, 1), 2), 1, 3);
assert(all(sampleCounts > 0), 'Each panel must contain evaluated examples.');

figureSize = options.FigureSize(:)';
fig = figure('Color', 'white', 'Units', 'pixels', ...
    'Position', [60, 60, figureSize], 'Renderer', 'painters');
% Reserve the outer margin for the final colorbar's scientific-notation label.
panelWidth = (figureSize(1) - 80) / 3;
side = min(panelWidth - 130, figureSize(2) - 210);
assert(side > 0, 'FigureSize is too small for three labeled matrices.');
plotBottom = 0.21 * figureSize(2);
colors = blue_colormap(256);
for panel = 1:3
    left = (panel - 1) * panelWidth + 100;
    position = [left / figureSize(1), plotBottom / figureSize(2), ...
        side / figureSize(1), side / figureSize(2)];
    ax = axes(fig, 'Units', 'normalized', 'Position', position, ...
        'XLim', [0.5, numClasses + 0.5], 'YLim', [0.5, numClasses + 0.5], ...
        'YDir', 'reverse', 'FontName', 'Times New Roman', 'FontSize', options.FontSize, ...
        'XColor', [0, 0, 0], 'YColor', [0, 0, 0], 'Color', 'white', ...
        'Box', 'on', 'Layer', 'top', 'LineWidth', 0.7, 'TickDir', 'in', ...
        'TickLength', [0.004, 0.004], 'Tag', 'confusion_triptych_axes', ...
        'UserData', struct('panelIndex', panel, 'panelName', panelNames(panel)));
    hold(ax, 'on');
    axis(ax, 'square');
    colormap(ax, colors);
    ax.CLim = limits(panel, :);
    for row = 1:numClasses
        for col = 1:numClasses
            value = counts(row, col, panel);
            colorIndex = min(size(colors, 1), ...
                max(1, fix(value / limits(panel, 2) * size(colors, 1)) + 1));
            rectangle(ax, 'Position', [col - 0.5, row - 0.5, 1, 1], ...
                'FaceColor', colors(colorIndex, :), 'EdgeColor', 'none', ...
                'Tag', 'confusion_triptych_values', 'UserData', value);
        end
    end
    ax.XTick = 1:numClasses;
    ax.YTick = 1:numClasses;
    ax.XTickLabel = classNames;
    ax.YTickLabel = classNames;
    xtickangle(ax, 45);
    xlabel(ax, 'Predicted modulation', 'FontName', 'Times New Roman', ...
        'FontSize', options.AxisLabelFontSize, 'Color', [0, 0, 0]);
    if panel == 1
        ylabel(ax, 'True modulation', 'FontName', 'Times New Roman', ...
            'FontSize', options.AxisLabelFontSize, 'Color', [0, 0, 0]);
    end
    title(ax, panelNames(panel), 'FontName', 'Times New Roman', ...
        'FontSize', options.TitleFontSize, 'FontWeight', 'normal', ...
        'Color', [0, 0, 0], 'Interpreter', 'none');
    for row = 1:numClasses
        for col = 1:numClasses
            value = counts(row, col, panel);
            textColor = [0, 0, 0];
            if value > 0.55 * limits(panel, 2)
                textColor = [1, 1, 1];
            end
            text(ax, col, row, sprintf('%d', value), ...
                'FontName', 'Times New Roman', 'FontSize', options.CellFontSize, ...
                'Color', textColor, 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'Interpreter', 'none', ...
                'Tag', 'confusion_triptych_annotation', 'UserData', value);
        end
    end
    colorbarPosition = [(left + side + 8) / figureSize(1), ...
        plotBottom / figureSize(2), 12 / figureSize(1), side / figureSize(2)];
    draw_count_colorbar(fig, colorbarPosition, colors, limits(panel, 2), ...
        options.FontSize, panel);
end
matrixData = struct('counts', counts, 'displayValues', counts, ...
    'normalization', 'none', 'classNames', classNames, 'panelNames', panelNames, ...
    'panelSampleCounts', sampleCounts, 'rowTotals', rowTotals, ...
    'colorLimits', limits, 'annotationCount', numel(counts), ...
    'axisMeaning', 'Rows: true classes; columns: predictions; colors and annotations: sample counts.');
for name = {'Source', 'Provenance', 'PanelIndices', 'Conditions'}
    if isfield(options, name{1})
        fieldName = lower_first(name{1});
        matrixData.(fieldName) = options.(name{1});
    end
end
fig.UserData = matrixData;
drawnow;
[outputDir, baseName] = fileparts(outputPath);
if ~isempty(outputDir) && ~isfolder(outputDir)
    mkdir(outputDir);
end
exportgraphics(fig, outputPath, 'Resolution', 300);
fprintf('Saved %s\n', outputPath);
formats = string(options.ExportFormats);
for format = formats(:)'
    path = fullfile(outputDir, [baseName, '.', char(lower(format))]);
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
            error('Unsupported export format: %s.', format);
    end
    fprintf('Saved %s\n', path);
end
end

function draw_count_colorbar(fig, position, colors, limit, fontSize, panel)
barAxes = axes(fig, 'Units', 'normalized', 'Position', position, ...
    'XLim', [0, 1], 'YLim', [0, limit], 'XTick', [], 'YDir', 'normal', ...
    'YAxisLocation', 'right', 'FontName', 'Times New Roman', 'FontSize', fontSize, ...
    'Color', 'white', 'XColor', [0, 0, 0], 'YColor', [0, 0, 0], ...
    'Box', 'on', 'Layer', 'top', 'LineWidth', 0.6, 'TickDir', 'in', ...
    'Tag', 'confusion_triptych_colorbar_axes', 'UserData', panel);
barAxes.YAxis.Exponent = max(0, floor(log10(limit)));
hold(barAxes, 'on');
step = limit / size(colors, 1);
for index = 1:size(colors, 1)
    rectangle(barAxes, 'Position', [0, (index - 1) * step, 1, step], ...
        'FaceColor', colors(index, :), 'EdgeColor', 'none', ...
        'Tag', 'confusion_triptych_colorbar_values', 'UserData', (index - 0.5) * step);
end
end

function colors = blue_colormap(numColors)
anchors = [1.00, 1.00, 1.00; 0.78, 0.88, 0.96; 0.36, 0.65, 0.82; 0.06, 0.28, 0.55];
colors = interp1(linspace(0, 1, size(anchors, 1)), anchors, linspace(0, 1, numColors), 'linear');
end

function options = apply_default(options, name, value)
if ~isfield(options, name) || isempty(options.(name))
    options.(name) = value;
end
end

function value = lower_first(value)
value(1) = lower(value(1));
end
