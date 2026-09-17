function plot_modulation_snr_accuracy(csvFile, outputPath, options)
%PLOT_MODULATION_SNR_ACCURACY Plot one SNR curve per modulation class.

arguments
    csvFile (1, :) char
    outputPath (1, :) char
    options struct = struct()
end

options = apply_default(options, 'Title', 'Per-Modulation Accuracy vs SNR');
options = apply_default(options, 'LegendColumns', 5);
if ~exist(csvFile, 'file')
    error('Result file not found: %s', csvFile);
end

data = readtable(csvFile, 'TextType', 'string');
required = {'modulation', 'snr', 'accuracy'};
if ~all(ismember(required, data.Properties.VariableNames))
    error('CSV must contain modulation, snr, and accuracy columns.');
end

classNames = unique(data.modulation, 'stable');
numClasses = numel(classNames);
colors = amr_class_colors(numClasses);
markers = {'o', 's', '^', 'd', 'v', '>', '<', 'p', 'h'};
fig = figure('Color', 'w', 'Position', [100, 80, 900, 650]);
ax = axes(fig);
hold(ax, 'on');
for classIdx = 1:numClasses
    mask = data.modulation == classNames(classIdx);
    snr = data.snr(mask);
    accuracy = data.accuracy(mask);
    [snr, order] = sort(snr);
    plot(ax, snr, accuracy(order), ...
        'Color', colors(classIdx, :), ...
        'Marker', markers{mod(classIdx - 1, numel(markers)) + 1}, ...
        'MarkerFaceColor', colors(classIdx, :), ...
        'MarkerEdgeColor', colors(classIdx, :), ...
        'LineWidth', 1.35, ...
        'MarkerSize', 4, ...
        'DisplayName', classNames(classIdx));
end

box(ax, 'on');
grid(ax, 'on');
ax.GridLineStyle = ':';
ax.GridAlpha = 0.35;
ax.FontName = 'Times New Roman';
ax.FontSize = 10;
xlabel(ax, 'SNR (dB)', 'FontName', 'Times New Roman');
ylabel(ax, 'Accuracy', 'FontName', 'Times New Roman');
title(ax, options.Title, 'FontName', 'Times New Roman', 'FontWeight', 'normal');
xlim(ax, [-20, 18]);
xticks(ax, -20:2:18);
ylim(ax, [0, 1]);
yticks(ax, 0:0.1:1);
legend(ax, 'Location', 'southeast', ...
    'NumColumns', min(options.LegendColumns, numClasses), ...
    'FontSize', 7, 'Box', 'on');

outputDir = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
exportgraphics(fig, outputPath, 'Resolution', 300);
fprintf('Saved per-modulation SNR figure to %s\n', outputPath);
end

function options = apply_default(options, name, value)
if ~isfield(options, name) || isempty(options.(name))
    options.(name) = value;
end
end
