function [fig, data] = plot_tsne_triptych(featureFiles, outputPath, options)
%PLOT_TSNE_TRIPTYCH Plot three independently computed feature projections.
% Each MAT file must contain features (N-by-48), zero-based labels,
% classNames, and targetSnr. Rows and labels are never resampled or reordered.
% Each panel resets RNG to the same fixed seed before its own t-SNE fit.
% Cache entries are keyed by the source MAT file SHA256 and every projection
% parameter, so changing only figure layout reuses the original embedding.
% PNG, vector PDF, editable FIG, and the embedding cache retain provenance.

arguments
    featureFiles
    outputPath (1, :) char
    options struct = struct()
end

files = string(featureFiles(:));
if numel(files) ~= 3 || any(ismissing(files) | strlength(files) == 0)
    error('Provide the Overall, 0 dB, and 10 dB feature files in that order.');
end
titles = string(option(options, 'PanelTitles', {'Overall', '0 dB', '10 dB'}));
titles = titles(:);
if numel(titles) ~= 3
    error('PanelTitles must contain exactly three labels.');
end
expectedSNR = [NaN, 0, 10];
seed = option(options, 'Seed', 42);
perplexity = option(options, 'Perplexity', 30);
expectedWidth = option(options, 'ExpectedFeatureDimensions', 48);
perClass = option(options, 'ExpectedSamplesPerClass', 300);
validateattributes(seed, {'numeric'}, {'scalar', 'integer', 'nonnegative', 'finite'});
validateattributes(perplexity, {'numeric'}, {'scalar', 'positive', 'finite'});
validateattributes([expectedWidth, perClass], {'numeric'}, ...
    {'vector', 'numel', 2, 'integer', 'positive', 'finite'});
[outDir, baseName] = fileparts(outputPath);
cacheFile = char(option(options, 'CacheFile', ...
    fullfile(outDir, '..', 'figure_data', 'saftformer_tsne_embeddings.mat')));
cacheVersion = uint32(1);
cacheEntries = empty_cache();
if isfile(cacheFile)
    existing = load(cacheFile, 'cacheVersion', 'cacheEntries');
    if isfield(existing, 'cacheVersion') && existing.cacheVersion == cacheVersion ...
            && isfield(existing, 'cacheEntries') && isstruct(existing.cacheEntries) ...
            && all(isfield(existing.cacheEntries, fieldnames(cacheEntries)))
        cacheEntries = existing.cacheEntries;
    end
end

embeddings = cell(3, 1);
labels = cell(3, 1);
featureHashes = strings(3, 1);
params = cell(3, 1);
sourceMetadata = cell(3, 1);
cacheReused = false(3, 1);
checkpointHashes = strings(3, 1);
classNames = strings(0, 1);
for panelIdx = 1:3
    path = char(files(panelIdx));
    if ~isfile(path)
        error('Feature file not found: %s', path);
    end
    featureHashes(panelIdx) = file_sha256(path);
    raw = load(path);
    required = {'features', 'labels', 'classNames', 'targetSnr'};
    if ~all(isfield(raw, required))
        error('Feature file must contain features, labels, classNames, and targetSnr: %s.', path);
    end
    features = double(raw.features);
    currentLabels = double(raw.labels(:));
    if ischar(raw.classNames) && size(raw.classNames, 1) > 1
        currentNames = string(cellstr(raw.classNames));
    else
        currentNames = string(raw.classNames(:));
    end
    currentNames = currentNames(:);
    validateattributes(features, {'numeric'}, ...
        {'2d', 'real', 'finite', 'ncols', expectedWidth});
    validateattributes(currentLabels, {'numeric'}, ...
        {'vector', 'real', 'finite', 'integer', '>=', 0, '<=', 14});
    if numel(currentNames) ~= 15 || numel(unique(currentNames)) ~= 15 ...
            || numel(currentLabels) ~= size(features, 1) ...
            || size(features, 1) ~= 15 * perClass
        error('Expected 15 classes with %d feature rows per class: %s.', perClass, path);
    end
    support = accumarray(currentLabels + 1, 1, [15, 1]);
    if any(support ~= perClass)
        error('Every class must contain exactly %d records: %s.', perClass, path);
    end
    if panelIdx == 1
        classNames = currentNames;
    elseif ~isequal(classNames, currentNames)
        error('All panels must use the same class-name order and label mapping.');
    end
    targetSnr = double(raw.targetSnr);
    if ~isscalar(targetSnr) || ~isequaln(targetSnr, expectedSNR(panelIdx))
        error('Feature file targetSnr does not match its panel: %s.', path);
    end
    if isfield(raw, 'snr')
        sampleSNR = double(raw.snr(:));
        if numel(sampleSNR) ~= size(features, 1) || any(~isfinite(sampleSNR))
            error('Per-record snr metadata must match the feature rows.');
        end
        if panelIdx > 1 && any(sampleSNR ~= expectedSNR(panelIdx))
            error('A fixed-SNR feature file contains another SNR: %s.', path);
        end
    end
    if isfield(raw, 'sourceIndices')
        indices = double(raw.sourceIndices(:));
        if numel(indices) ~= size(features, 1) || numel(unique(indices)) ~= numel(indices)
            error('sourceIndices must identify one distinct source record per feature row.');
        end
    end
    if isfield(raw, 'checkpointHash')
        checkpointHashes(panelIdx) = string(raw.checkpointHash);
    elseif isfield(raw, 'checkpointSha256')
        checkpointHashes(panelIdx) = string(raw.checkpointSha256);
    end
    sourceMetadata{panelIdx} = rmfield(raw, {'features', 'labels'});
    params{panelIdx} = struct('seed', seed, 'perplexity', perplexity, ...
        'numDimensions', 2, ...
        'numPCAComponents', min([50, size(features, 2), size(features, 1) - 1]), ...
        'standardize', true, 'algorithm', 'barneshut', 'distance', 'euclidean', ...
        'maxIterations', 1000, 'tolerance', 1e-10, ...
        'matlabRelease', version('-release'), 'numObservations', size(features, 1), ...
        'featureDimensions', size(features, 2));
    if perplexity >= size(features, 1)
        error('Perplexity must be smaller than the number of feature rows.');
    end
    currentParams = params{panelIdx};
    match = panelIdx <= numel(cacheEntries) ...
        && strcmp(cacheEntries(panelIdx).featureSha256, char(featureHashes(panelIdx))) ...
        && isequaln(cacheEntries(panelIdx).params, currentParams) ...
        && isequal(cacheEntries(panelIdx).labels, currentLabels) ...
        && isequal(cacheEntries(panelIdx).classNames, classNames) ...
        && isequal(size(cacheEntries(panelIdx).embedding), [size(features, 1), 2]) ...
        && all(isfinite(cacheEntries(panelIdx).embedding(:)));
    if match
        embeddings{panelIdx} = cacheEntries(panelIdx).embedding;
        cacheReused(panelIdx) = true;
        fprintf('Reusing cached t-SNE: %s\n', titles(panelIdx));
    else
        if exist('tsne', 'file') ~= 2
            error('Computing t-SNE requires MATLAB Statistics and Machine Learning Toolbox.');
        end
        rng(seed, 'twister');
        fprintf('Computing independent t-SNE: %s (%d rows x %d features).\n', ...
            titles(panelIdx), size(features, 1), size(features, 2));
        optimization = statset('MaxIter', currentParams.maxIterations, ...
            'TolFun', currentParams.tolerance);
        embeddings{panelIdx} = tsne(features, ...
            'NumDimensions', currentParams.numDimensions, ...
            'NumPCAComponents', currentParams.numPCAComponents, ...
            'Perplexity', currentParams.perplexity, ...
            'Standardize', currentParams.standardize, ...
            'Algorithm', currentParams.algorithm, 'Distance', currentParams.distance, ...
            'Options', optimization, 'Verbose', 0);
        if ~isequal(size(embeddings{panelIdx}), [size(features, 1), 2]) ...
                || any(~isfinite(embeddings{panelIdx}), 'all')
            error('t-SNE returned invalid or missing feature rows.');
        end
    end
    labels{panelIdx} = currentLabels;
    entry = struct('featureFile', path, 'featureSha256', char(featureHashes(panelIdx)), ...
        'params', currentParams, 'embedding', embeddings{panelIdx}, ...
        'labels', currentLabels, 'classNames', classNames, ...
        'targetSnr', targetSnr, 'condition', char(titles(panelIdx)), ...
        'sourceMetadata', sourceMetadata{panelIdx});
    cacheEntries(panelIdx) = entry;
    cacheDir = fileparts(cacheFile);
    if ~isempty(cacheDir) && ~exist(cacheDir, 'dir')
        mkdir(cacheDir);
    end
    % Save each completed panel so an interrupted run can resume from cache.
    save(cacheFile, 'cacheVersion', 'cacheEntries', '-v7');
end
nonemptyHashes = checkpointHashes(strlength(checkpointHashes) > 0);
if ~isempty(nonemptyHashes) && numel(unique(nonemptyHashes)) ~= 1
    error('The three feature files must originate from the same checkpoint.');
end

figureSize = option(options, 'FigureSize', [1600, 760]);
markerSize = option(options, 'MarkerSize', 8);
titleFontSize = option(options, 'TitleFontSize', 22);
legendFontSize = option(options, 'LegendFontSize', 18);
validateattributes(figureSize, {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 2});
validateattributes([markerSize, titleFontSize, legendFontSize], {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 3});
colors = amr_class_colors(15);
fig = figure('Color', 'white', 'Units', 'pixels', ...
    'Position', [60, 60, figureSize(:)'], 'Renderer', 'painters');
panelWidth = 0.30;
panelHeight = panelWidth * figureSize(1) / figureSize(2);
lefts = [0.025, 0.35, 0.675];
axesHandles = gobjects(3, 1);
legendHandles = gobjects(15, 1);
for panelIdx = 1:3
    ax = axes(fig, 'Units', 'normalized', ...
        'Position', [lefts(panelIdx), 0.29, panelWidth, panelHeight], ...
        'FontName', 'Times New Roman', 'XTick', [], 'YTick', [], ...
        'XColor', 'black', 'YColor', 'black', 'LineWidth', 0.8, ...
        'Box', 'on', 'Layer', 'top', 'Tag', 'tsne_triptych_axes');
    axesHandles(panelIdx) = ax;
    hold(ax, 'on');
    embedding = embeddings{panelIdx};
    for classIdx = 1:15
        selected = labels{panelIdx} == classIdx - 1;
        dots = scatter(ax, embedding(selected, 1), embedding(selected, 2), markerSize, ...
            'Marker', 'o', 'MarkerFaceColor', colors(classIdx, :), ...
            'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 1, 'MarkerEdgeAlpha', 1, ...
            'DisplayName', classNames(classIdx), 'Tag', 'tsne_triptych_class');
        dots.UserData = struct('classId', classIdx - 1, 'condition', titles(panelIdx));
        if panelIdx == 1
            legendHandles(classIdx) = dots;
        end
    end
    % Every panel gets its own square, equal-unit view. No cross-panel limits
    % or coordinate alignment are imposed on the independent projections.
    lower = min(embedding, [], 1);
    upper = max(embedding, [], 1);
    center = (lower + upper) / 2;
    span = max(max(upper - lower), eps) * 1.08;
    xlim(ax, center(1) + [-0.5, 0.5] * span);
    ylim(ax, center(2) + [-0.5, 0.5] * span);
    ax.DataAspectRatio = [1, 1, 1];
    ax.PlotBoxAspectRatio = [1, 1, 1];
    title(ax, titles(panelIdx), 'FontName', 'Times New Roman', ...
        'FontSize', titleFontSize, 'FontWeight', 'normal', ...
        'Color', 'black', 'Interpreter', 'none');
end
lgd = legend(axesHandles(1), legendHandles, cellstr(classNames), ...
    'NumColumns', 5, 'Location', 'southoutside', 'FontName', 'Times New Roman', ...
    'FontSize', legendFontSize, 'TextColor', 'black', 'Box', 'on', ...
    'Interpreter', 'none', 'AutoUpdate', 'off');
lgd.ItemTokenSize = [16, 10];
lgd.Units = 'normalized';
lgd.Position = [0.12, 0.055, 0.76, 0.145];
for panelIdx = 1:3
    axesHandles(panelIdx).Position = [lefts(panelIdx), 0.29, panelWidth, panelHeight];
end
data = struct('featureFiles', files, 'featureSha256', featureHashes, ...
    'embeddings', {embeddings}, 'labels', {labels}, 'classNames', classNames, ...
    'conditions', titles, 'targetSnr', expectedSNR, 'params', {params}, ...
    'sourceMetadata', {sourceMetadata}, 'checkpointSha256', checkpointHashes, ...
    'cacheFile', cacheFile, 'cacheReused', cacheReused, ...
    'colors', colors, 'independentProjections', true);
fig.UserData = data;
drawnow;
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end
exportgraphics(fig, outputPath, 'Resolution', 300, 'BackgroundColor', 'white');
exportgraphics(fig, fullfile(outDir, [baseName, '.pdf']), ...
    'ContentType', 'vector', 'BackgroundColor', 'white');
savefig(fig, fullfile(outDir, [baseName, '.fig']));
fprintf('Saved t-SNE triptych: %s (.png, .pdf, .fig).\n', fullfile(outDir, baseName));
end

function entries = empty_cache()
entries = struct('featureFile', {}, 'featureSha256', {}, 'params', {}, ...
    'embedding', {}, 'labels', {}, 'classNames', {}, 'targetSnr', {}, ...
    'condition', {}, 'sourceMetadata', {});
end

function hash = file_sha256(path)
fid = fopen(path, 'rb');
if fid < 0
    error('Could not open feature file: %s', path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
digest = java.security.MessageDigest.getInstance('SHA-256');
while ~feof(fid)
    bytes = fread(fid, 1024 * 1024, '*uint8');
    if ~isempty(bytes)
        digest.update(typecast(bytes, 'int8'));
    end
end
hash = lower(sprintf('%02x', typecast(digest.digest(), 'uint8')));
end

function value = option(options, name, default)
if isfield(options, name) && ~isempty(options.(name))
    value = options.(name);
else
    value = default;
end
end
