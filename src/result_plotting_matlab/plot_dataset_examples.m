function [fig, provenance] = plot_dataset_examples(datasetPath, outputPath)
%PLOT_DATASET_EXAMPLES Plot three real, independently generated I/Q examples.
% Reads only metadata, three 1000-entry split blocks, and three 2x512 signals.
% Selection is fixed before viewing: QPSK, 10 dB, AWGN / DoublySelective /
% 3GPP_TDL_C, first test example in each cell. The panels are independent
% realizations, not the same transmitted waveform passed through channels.
% Sample indices and raw signals are preserved in the editable FIG UserData.

if ~isfile(datasetPath)
    error('Dataset file not found: %s.', datasetPath);
end
datasetPath = char(java.io.File(datasetPath).getCanonicalPath());
m = matfile(datasetPath);
if ~logical(m.generationComplete)
    error('Refusing to read an incompletely generated dataset.');
end
meta = m.meta;
if ~strcmp(meta.sampleOrder, 'class -> channel -> SNR -> sample')
    error('Expected explicit class/channel/SNR/sample ordering metadata.');
end
rawClassNames = m.classNames;
rawChannelNames = m.channelNames;
classNames = cellstr(string(rawClassNames(:)));
channelNames = cellstr(string(rawChannelNames(:)));
modulationName = 'QPSK';
snrDb = 10;
requestedChannels = {'AWGN', 'DoublySelective', '3GPP_TDL_C'};
displayNames = {'AWGN', 'Doubly selective', '3GPP TDL-C'};
classIndex = find(strcmp(classNames, modulationName), 1);
snrValues = double(meta.snrs(:));
snrIndex = find(snrValues == snrDb, 1);
if isempty(classIndex) || isempty(snrIndex)
    error('The requested modulation or SNR is absent from dataset metadata.');
end
numPerCell = double(meta.samplesPerClassPerSnrPerChannel);
numChannels = numel(channelNames);
numSnrs = numel(snrValues);
signalLength = double(meta.signalLength);
numExamples = numel(requestedChannels);
samples = zeros(numExamples, 2, signalLength, 'single');
examples = repmat(struct('channel', '', 'modulation', modulationName, ...
    'snrDb', snrDb, 'matlabSampleIndex', 0, 'zeroBasedSampleIndex', 0, ...
    'cellFirstMatlabIndex', 0, 'cellLastMatlabIndex', 0, ...
    'offsetWithinCellOneBased', 0, 'classLabel', 0, 'channelLabel', 0, ...
    'splitLabel', 0, 'earlierTestExamplesInCell', 0), numExamples, 1);

for exampleIdx = 1:numExamples
    channelIndex = find(strcmp(channelNames, requestedChannels{exampleIdx}), 1);
    if isempty(channelIndex)
        error('Unknown channel: %s.', requestedChannels{exampleIdx});
    end
    cellIndex = ((classIndex - 1) * numChannels + channelIndex - 1) ...
        * numSnrs + snrIndex - 1;
    firstIndex = cellIndex * numPerCell + 1;
    lastIndex = firstIndex + numPerCell - 1;
    cellSplit = m.split(firstIndex:lastIndex, 1);
    firstTestOffset = find(cellSplit == 2, 1, 'first');
    if isempty(firstTestOffset)
        error('No held-out test example in the requested dataset cell.');
    end
    sampleIndex = firstIndex + firstTestOffset - 1;
    actualClass = double(m.y(sampleIndex, 1));
    actualChannel = double(m.channel(sampleIndex, 1));
    actualSnr = double(m.snr(sampleIndex, 1));
    actualSplit = double(m.split(sampleIndex, 1));
    assert(actualClass == classIndex - 1 && actualChannel == channelIndex - 1 ...
        && actualSnr == snrDb && actualSplit == 2, ...
        'Computed sample indices do not match the stored labels.');
    xi = m.X(sampleIndex, :, :);
    assert(isequal(size(xi), [1, 2, signalLength]) && all(isfinite(xi(:))), ...
        'Unexpected signal dimensions or non-finite dataset values.');
    samples(exampleIdx, :, :) = xi;
    examples(exampleIdx) = struct('channel', requestedChannels{exampleIdx}, ...
        'modulation', modulationName, 'snrDb', snrDb, ...
        'matlabSampleIndex', sampleIndex, 'zeroBasedSampleIndex', sampleIndex - 1, ...
        'cellFirstMatlabIndex', firstIndex, 'cellLastMatlabIndex', lastIndex, ...
        'offsetWithinCellOneBased', firstTestOffset, 'classLabel', actualClass, ...
        'channelLabel', actualChannel, 'splitLabel', actualSplit, ...
        'earlierTestExamplesInCell', sum(cellSplit(1:firstTestOffset-1) == 2));
end
assert(signalLength == 512, 'Expected 512-sample project signals.');
maxAmplitude = double(max(abs(samples(:))));
amplitudeLimit = max(0.5, ceil(maxAmplitude * 1.06 * 2) / 2);
colors = [68, 119, 170; 68, 170, 153] / 255;
source = dir(datasetPath);
provenance = struct('datasetFile', datasetPath, 'sourceFileBytes', source.bytes, ...
    'sourceFileModified', source.date, 'generationComplete', true, ...
    'generatedSamples', double(m.generatedSamples), ...
    'datasetSeed', double(meta.seed), 'sampleOrder', meta.sampleOrder, ...
    'selectionRule', 'First stored test example within each requested cell', ...
    'independentRealizations', true, 'signalLength', signalLength, ...
    'modulation', modulationName, 'snrDb', snrDb, ...
    'figurePixels', [1200, 480], 'sharedYLimits', [-amplitudeLimit, amplitudeLimit], ...
    'traceColorsRGB', colors, 'examples', examples);

fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'pixels', ...
    'Position', [80, 80, 1200, 480], 'Renderer', 'painters');
layout = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
axesHandles = gobjects(numExamples, 1);
traceHandles = gobjects(2, 1);
for exampleIdx = 1:numExamples
    ax = nexttile(layout);
    axesHandles(exampleIdx) = ax;
    hold(ax, 'on');
    for componentIdx = 1:2
        trace = plot(ax, 1:signalLength, ...
            reshape(samples(exampleIdx, componentIdx, :), 1, []), ...
            'Color', colors(componentIdx, :), 'LineWidth', 0.85);
        if exampleIdx == 1
            traceHandles(componentIdx) = trace;
        end
    end
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 16, ...
        'XColor', 'k', 'YColor', 'k', 'LineWidth', 0.7, ...
        'TickDir', 'in', 'Box', 'on', 'GridColor', [0, 0, 0], ...
        'GridAlpha', 0.12, 'GridLineStyle', ':', 'Layer', 'bottom');
    xlim(ax, [1, signalLength]);
    ylim(ax, [-amplitudeLimit, amplitudeLimit]);
    xticks(ax, [1, 128, 256, 384, 512]);
    yticks(ax, [-amplitudeLimit, 0, amplitudeLimit]);
    grid(ax, 'on');
    title(ax, sprintf('(%c) %s', char('a' + exampleIdx - 1), displayNames{exampleIdx}), ...
        'FontName', 'Times New Roman', 'FontSize', 17, 'FontWeight', 'normal', ...
        'Color', 'k', 'Interpreter', 'none');
    if exampleIdx < numExamples
        ax.XTickLabel = [];
    end
end
xlabel(layout, 'Sample index', 'FontName', 'Times New Roman', ...
    'FontSize', 18, 'Color', 'k');
ylabel(layout, 'Normalized amplitude', 'FontName', 'Times New Roman', ...
    'FontSize', 18, 'Color', 'k');
lgd = legend(axesHandles(1), traceHandles, {'I', 'Q'}, ...
    'FontName', 'Times New Roman', 'FontSize', 17, 'TextColor', 'k', ...
    'Orientation', 'horizontal', 'Box', 'off', 'AutoUpdate', 'off');
lgd.Layout.Tile = 'north';
fig.UserData = struct('provenance', provenance, 'samples', samples, ...
    'matlabSampleIDs', [examples.matlabSampleIndex], ...
    'zeroBasedSampleIDs', [examples.zeroBasedSampleIndex]);
drawnow;

outputDir = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
[folder, baseName] = fileparts(outputPath);
exportgraphics(fig, outputPath, 'Resolution', 300, 'BackgroundColor', 'white');
exportgraphics(fig, fullfile(folder, [baseName, '.pdf']), ...
    'ContentType', 'vector', 'BackgroundColor', 'white');
savefig(fig, fullfile(folder, [baseName, '.fig']));
end
