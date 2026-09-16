function visualize_amr_dataset(datasetPath, outputDir, maxExamples, targetSnr, targetChannels)
%VISUALIZE_AMR_DATASET Plot paper-style I/Q time-domain examples.
%
% Usage:
%   visualize_amr_dataset('../../datasets/matlab_amr_dataset.mat', 'runs/matlab_dataset_viz')

if nargin < 1 || isempty(datasetPath)
    cfg = amr_dataset_config();
    datasetPath = cfg.outputPath;
end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(fileparts(mfilename('fullpath')), 'runs', 'matlab_dataset_viz');
end
if nargin < 3 || isempty(maxExamples)
    maxExamples = inf;
end
if nargin < 4 || isempty(targetSnr)
    targetSnr = 10;
end
if nargin < 5
    targetChannels = {};
end
if ~exist(datasetPath, 'file')
    error('Dataset file not found: %s', datasetPath);
end
if ~isscalar(maxExamples) || maxExamples <= 0
    error('maxExamples must be a positive scalar.');
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

m = matfile(datasetPath);
variables = who(m);
if ismember('generationComplete', variables) && ~logical(m.generationComplete)
    error(['Dataset generation is incomplete: %s\nRegenerate the dataset before ' ...
        'creating figures.'], datasetPath);
end
classNames = normalize_cell_names(m.classNames);
channelNames = normalize_cell_names(m.channelNames);
meta = m.meta;

if ~isfield(meta, 'sampleOrder') || ...
        ~strcmp(meta.sampleOrder, 'class -> channel -> SNR -> sample')
    error('Dataset does not provide the sample-order metadata required for indexed visualization.');
end

snrValues = double(meta.snrs(:).');
[snrDistance, targetSnrIdx] = min(abs(snrValues - targetSnr));
selectedSnr = snrValues(targetSnrIdx);
if snrDistance > 0
    warning('Requested SNR %g dB is unavailable; using nearest value %g dB.', ...
        targetSnr, selectedSnr);
end
numPerCell = double(meta.samplesPerClassPerSnrPerChannel);
numChannels = numel(channelNames);
numSnrs = numel(snrValues);

if isempty(targetChannels)
    selectedChannelIds = 0:(numel(channelNames) - 1);
else
    if ischar(targetChannels) || isstring(targetChannels)
        targetChannels = cellstr(targetChannels);
    end
    selectedChannelIds = [];
    for i = 1:numel(targetChannels)
        channelId = find(strcmpi(channelNames, targetChannels{i}), 1) - 1;
        if isempty(channelId)
            warning('Skipping unknown channel: %s', targetChannels{i});
        else
            selectedChannelIds(end + 1) = channelId; %#ok<AGROW>
        end
    end
end

for selectedIdx = 1:numel(selectedChannelIds)
    channelId = selectedChannelIds(selectedIdx);
    channelName = channelNames{channelId + 1};
    numClasses = min(numel(classNames), maxExamples);
    sampleIds = zeros(1, numClasses);
    sampleLabels = cell(1, numClasses);
    for classIdx = 1:numClasses
        cellIndex = ((classIdx - 1) * numChannels + channelId) * numSnrs + ...
            (targetSnrIdx - 1);
        sampleOffset = floor((numPerCell - 1) / 2);
        sampleIds(classIdx) = cellIndex * numPerCell + sampleOffset + 1;
        sampleLabels{classIdx} = sprintf('%s %gdB SNR', ...
            classNames{classIdx}, selectedSnr);
    end

    verify_sample_labels(m, sampleIds, 0:(numClasses - 1), ...
        channelId, selectedSnr);
    outputName = sprintf('iq_time_examples_%s.png', safe_filename(channelName));
    plot_iq_examples(m, sampleIds, sampleLabels, outputDir, outputName);
end

function verify_sample_labels(m, sampleIds, expectedLabels, channelId, selectedSnr)
actualLabels = double(m.y(sampleIds, 1)).';
actualChannels = double(m.channel(sampleIds, 1)).';
actualSnrs = double(m.snr(sampleIds, 1)).';
assert(isequal(actualLabels, expectedLabels), ...
    'Computed visualization indices do not match modulation labels.');
assert(all(actualChannels == channelId), ...
    'Computed visualization indices do not match the selected channel.');
assert(all(actualSnrs == selectedSnr), ...
    'Computed visualization indices do not match the selected SNR.');
end
fprintf('Saved I/Q time-domain examples to %s\n', outputDir);
end

function names = normalize_cell_names(raw)
if iscell(raw)
    names = raw(:).';
else
    names = cellstr(raw);
end
for idx = 1:numel(names)
    names{idx} = strtrim(char(names{idx}));
end
end

function fileName = safe_filename(text)
fileName = regexprep(text, '[^A-Za-z0-9]+', '_');
fileName = regexprep(fileName, '^_+|_+$', '');
end

function plot_iq_examples(m, sampleIds, sampleLabels, outputDir, outputName)
n = numel(sampleIds);
cols = min(5, n);
rows = ceil(n / cols);
firstSample = squeeze(m.X(sampleIds(1), :, :));
signalLength = size(firstSample, 2);
samples = zeros(n, 2, signalLength, 'single');
samples(1, :, :) = firstSample;
for idx = 2:n
    samples(idx, :, :) = m.X(sampleIds(idx), :, :);
end
amplitudeLimit = max(abs(samples(:)));
if ~isfinite(amplitudeLimit) || amplitudeLimit < eps('single')
    amplitudeLimit = 1;
else
    amplitudeLimit = 1.05 * amplitudeLimit;
end

fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, cols * 260, rows * 190 + 55]);
layout = tiledlayout(rows, cols, 'TileSpacing', 'compact', 'Padding', 'compact');
xlabel(layout, 'Sample index', 'FontSize', 10);
ylabel(layout, 'Normalized amplitude', 'FontSize', 10);

for i = 1:n
    nexttile;
    xi = squeeze(samples(i, :, :));
    plot(xi(1, :), 'Color', [0, 0.32, 0.78], 'LineWidth', 0.85);
    hold on;
    plot(xi(2, :), 'Color', [0, 0.55, 0.18], 'LineWidth', 0.85);
    hold off;
    title(sampleLabels{i}, 'FontSize', 8, 'Interpreter', 'none');
    xlim([1, size(xi, 2)]);
    ylim([-amplitudeLimit, amplitudeLimit]);
    box on;
    grid on;
    set(gca, 'FontSize', 7, 'LineWidth', 0.6);
end
for i = (n + 1):(rows * cols)
    nexttile;
    axis off;
end

outputPath = fullfile(outputDir, outputName);
exportgraphics(fig, outputPath, 'Resolution', 300);
close(fig);
end
