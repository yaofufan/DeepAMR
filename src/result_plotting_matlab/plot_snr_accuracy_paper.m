function plot_snr_accuracy_paper(experiments, outputPath)
%PLOT_SNR_ACCURACY_PAPER Plot paper-style SNR accuracy comparison curves.
%
% Usage:
%   exp(1).title = 'MATLAB AMR Dataset';
%   exp(1).metricsFiles = {'../runs/eval_oshea_resnet/snr_accuracy.csv'};
%   exp(1).modelNames = {'ResNet'};
%   exp(1).insetXLim = [0 18];
%   exp(1).insetYLim = [0.65 0.90];
%   plot_snr_accuracy_paper(exp, 'figures/snr_accuracy_paper.png');
%
% For multi-model comparison, pass one snr_accuracy.csv file per model.
% Each CSV file must be produced by python -m amc.evaluate.

if nargin < 1 || isempty(experiments)
    experiments = default_experiment();
end
if nargin < 2 || isempty(outputPath)
    outputPath = fullfile('figures', 'snr_accuracy_paper.png');
end

if ~isstruct(experiments)
    error('experiments must be a struct array.');
end

outDir = fileparts(outputPath);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end

numPanels = numel(experiments);
colors = lines(8);
markers = {'o', 's', '^', 'd', 'v', 'p', 'h', 'x'};

figWidth = max(560, 520 * numPanels);
figHeight = 560;
fig = figure('Color', 'w', 'Position', [80, 80, figWidth, figHeight]);
tiledlayout(1, numPanels, 'TileSpacing', 'compact', 'Padding', 'compact');

for panelIdx = 1:numPanels
    ax = nexttile;
    hold(ax, 'on');
    box(ax, 'on');
    grid(ax, 'on');
    ax.GridLineStyle = ':';
    ax.GridAlpha = 0.35;
    ax.LineWidth = 0.8;
    ax.FontName = 'Times New Roman';
    ax.FontSize = 10;

    expCfg = experiments(panelIdx);
    metricsFiles = as_cell(expCfg.metricsFiles);
    modelNames = as_cell(expCfg.modelNames);
    if numel(modelNames) ~= numel(metricsFiles)
        error('modelNames and metricsFiles must have the same length.');
    end

    snrCell = cell(numel(metricsFiles), 1);
    accCell = cell(numel(metricsFiles), 1);
    for modelIdx = 1:numel(metricsFiles)
        [snrValues, accuracy] = read_snr_accuracy(metricsFiles{modelIdx});
        snrCell{modelIdx} = snrValues;
        accCell{modelIdx} = accuracy;
        plot(ax, snrValues, accuracy, ...
            'Color', colors(mod(modelIdx - 1, size(colors, 1)) + 1, :), ...
            'Marker', markers{mod(modelIdx - 1, numel(markers)) + 1}, ...
            'LineWidth', 1.4, ...
            'MarkerSize', 4.5, ...
            'DisplayName', modelNames{modelIdx});
    end

    xlabel(ax, 'SNR (dB)', 'FontName', 'Times New Roman');
    ylabel(ax, 'Accuracy', 'FontName', 'Times New Roman');
    title(ax, subplot_title(panelIdx, expCfg.title), 'FontName', 'Times New Roman');
    xlim(ax, [-20, 18]);
    xticks(ax, -20:2:18);
    ylim(ax, [0, 1]);
    pbaspect(ax, [1, 1, 1]);
    legend(ax, 'Location', 'northwest', 'FontSize', 8, 'Box', 'on');

    if isfield(expCfg, 'insetXLim') && isfield(expCfg, 'insetYLim') ...
            && ~isempty(expCfg.insetXLim) && ~isempty(expCfg.insetYLim)
        draw_inset(ax, snrCell, accCell, modelNames, expCfg.insetXLim, expCfg.insetYLim, colors, markers);
    end
end

exportgraphics(fig, outputPath, 'Resolution', 300);
fprintf('Saved paper-style SNR figure to %s\n', outputPath);
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

function value = as_cell(value)
if ischar(value) || isstring(value)
    value = cellstr(value);
end
end

function label = subplot_title(panelIdx, textLabel)
letters = 'abcdefghijklmnopqrstuvwxyz';
if panelIdx <= numel(letters)
    label = sprintf('(%s) %s', letters(panelIdx), textLabel);
else
    label = sprintf('(%d) %s', panelIdx, textLabel);
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

[snrValues, order] = sort(snrValues(:));
accuracy = accuracy(order);
end

function draw_inset(parentAx, snrCell, accCell, modelNames, insetXLim, insetYLim, colors, markers)
parentPos = parentAx.Position;
insetPos = [
    parentPos(1) + parentPos(3) * 0.56, ...
    parentPos(2) + parentPos(4) * 0.10, ...
    parentPos(3) * 0.39, ...
    parentPos(4) * 0.38
];
insetAx = axes('Position', insetPos);
hold(insetAx, 'on');
box(insetAx, 'on');
grid(insetAx, 'on');
insetAx.GridLineStyle = ':';
insetAx.GridAlpha = 0.25;
insetAx.FontName = 'Times New Roman';
insetAx.FontSize = 7;
for modelIdx = 1:numel(modelNames)
    plot(insetAx, snrCell{modelIdx}, accCell{modelIdx}, ...
        'Color', colors(mod(modelIdx - 1, size(colors, 1)) + 1, :), ...
        'Marker', markers{mod(modelIdx - 1, numel(markers)) + 1}, ...
        'LineWidth', 1.0, ...
        'MarkerSize', 3.2);
end
xlim(insetAx, insetXLim);
xticks(insetAx, insetXLim(1):2:insetXLim(2));
ylim(insetAx, insetYLim);
end
