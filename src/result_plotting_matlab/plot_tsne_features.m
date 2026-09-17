function embedding = plot_tsne_features(featureFile, outputPath, options)
%PLOT_TSNE_FEATURES Plot learned PyTorch features with MATLAB t-SNE.

arguments
    featureFile (1, :) char
    outputPath (1, :) char
    options struct = struct()
end

options = apply_default(options, 'Seed', 42);
options = apply_default(options, 'Perplexity', 30);
options = apply_default(options, 'MarkerSize', 12);

if exist('tsne', 'file') ~= 2
    error('tsne requires MATLAB Statistics and Machine Learning Toolbox.');
end
if ~exist(featureFile, 'file')
    error('Feature file not found: %s', featureFile);
end

data = load(featureFile);
required = {'features', 'labels', 'classNames', 'targetSnr'};
for idx = 1:numel(required)
    if ~isfield(data, required{idx})
        error('Feature file is missing variable: %s', required{idx});
    end
end

features = double(data.features);
labels = double(data.labels(:));
classNames = cellstr(string(data.classNames(:)));
targetSnr = double(data.targetSnr(1));
if size(features, 1) ~= numel(labels)
    error('features and labels contain different sample counts.');
end

rng(options.Seed, 'twister');
numPcaComponents = min([50, size(features, 2), size(features, 1) - 1]);
perplexity = min(options.Perplexity, floor((size(features, 1) - 1) / 3));
embedding = tsne(features, ...
    'NumDimensions', 2, ...
    'NumPCAComponents', numPcaComponents, ...
    'Perplexity', perplexity, ...
    'Standardize', true);

outputDir = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

numClasses = numel(classNames);
colors = amr_class_colors(numClasses);
fig = figure('Color', 'w', 'Position', [100, 80, 780, 720]);
ax = axes(fig);
hold(ax, 'on');
handles = gobjects(numClasses, 1);
for classId = 0:numClasses - 1
    mask = labels == classId;
    handles(classId + 1) = scatter(ax, ...
        embedding(mask, 1), embedding(mask, 2), options.MarkerSize, ...
        'Marker', 'o', ...
        'MarkerFaceColor', colors(classId + 1, :), ...
        'MarkerEdgeColor', colors(classId + 1, :), ...
        'MarkerFaceAlpha', 0.72, ...
        'MarkerEdgeAlpha', 0.72);
end

box(ax, 'on');
axis(ax, 'square');
ax.FontName = 'Times New Roman';
ax.FontSize = 10;
ax.XTick = [];
ax.YTick = [];
title(ax, sprintf('t-SNE Visualization (SNR = %gdB)', targetSnr), ...
    'FontName', 'Times New Roman', 'FontWeight', 'normal');
legend(ax, handles, classNames, ...
    'Location', 'southeast', ...
    'NumColumns', min(5, numClasses), ...
    'FontSize', 7, ...
    'Box', 'on');

exportgraphics(fig, outputPath, 'Resolution', 300);
save(fullfile(outputDir, sprintf('tsne_embedding_%gdB.mat', targetSnr)), ...
    'embedding', 'labels', 'classNames', 'targetSnr');
fprintf('Saved t-SNE figure to %s\n', outputPath);
end

function options = apply_default(options, name, value)
if ~isfield(options, name) || isempty(options.(name))
    options.(name) = value;
end
end
