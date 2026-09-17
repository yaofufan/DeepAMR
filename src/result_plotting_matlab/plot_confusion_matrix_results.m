function plot_confusion_matrix_results(csvFile, outputPath, options)
%PLOT_CONFUSION_MATRIX_RESULTS Plot a confusion matrix using sample counts.

arguments
    csvFile (1, :) char
    outputPath (1, :) char
    options struct = struct()
end

options = apply_default(options, 'Title', 'Confusion Matrix');
if ~exist(csvFile, 'file')
    error('Result file not found: %s', csvFile);
end

raw = readcell(csvFile);
classNames = cellstr(string(raw(1, 2:end)));
rowNames = cellstr(string(raw(2:end, 1)));
counts = cell2mat(raw(2:end, 2:end));
if ~isequal(classNames(:), rowNames(:))
    error('Confusion-matrix row and column class names do not match.');
end
maxCount = max(counts, [], 'all');
colorLimit = max(maxCount, 1);

numClasses = numel(classNames);
figSize = max(680, 45 * numClasses);
fig = figure('Color', 'w', 'Position', [100, 60, figSize, figSize]);
ax = axes(fig);
imagesc(ax, counts, [0, colorLimit]);
axis(ax, 'square');
colormap(ax, blue_colormap(256));
colorbar(ax);
ax.FontName = 'Times New Roman';
ax.FontSize = 9;
ax.XTick = 1:numClasses;
ax.YTick = 1:numClasses;
ax.XTickLabel = classNames;
ax.YTickLabel = classNames;
xtickangle(ax, 45);
xlabel(ax, 'Predicted Modulation', 'FontName', 'Times New Roman');
ylabel(ax, 'True Modulation', 'FontName', 'Times New Roman');
title(ax, options.Title, 'FontName', 'Times New Roman', 'FontWeight', 'normal');

for row = 1:numClasses
    for col = 1:numClasses
        textColor = [0, 0, 0];
        if counts(row, col) > 0.55 * colorLimit
            textColor = [1, 1, 1];
        end
        text(ax, col, row, sprintf('%d', counts(row, col)), ...
            'HorizontalAlignment', 'center', ...
            'FontName', 'Times New Roman', ...
            'FontSize', 7, ...
            'Color', textColor);
    end
end

outputDir = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
exportgraphics(fig, outputPath, 'Resolution', 300);
fprintf('Saved confusion matrix to %s\n', outputPath);
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

function options = apply_default(options, name, value)
if ~isfield(options, name) || isempty(options.(name))
    options.(name) = value;
end
end
