function [fig, provenance] = plot_dataset_overview(datasetPath, outputPath)
%PLOT_DATASET_OVERVIEW Show all 15 modulation types using actual held-out data.
% Fixed selection: dataset class order, AWGN, 18 dB, first stored test example
% within every modulation/channel/SNR cell. Plot the first 128 of 512 stored
% samples, without waveform regeneration or rescaling. All panels share the
% same amplitude scale. FIG UserData retains full and plotted raw I/Q data.

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
    error('Dataset ordering metadata is required for indexed visualization.');
end
rawClassNames = m.classNames;
rawChannelNames = m.channelNames;
classNames = cellstr(string(rawClassNames(:)));
channelNames = cellstr(string(rawChannelNames(:)));
numClasses = numel(classNames);
numChannels = numel(channelNames);
snrValues = double(meta.snrs(:));
numSnrs = numel(snrValues);
numPerCell = double(meta.samplesPerClassPerSnrPerChannel);
signalLength = double(meta.signalLength);
channelName = 'AWGN';
snrDb = 18;
channelIndex = find(strcmp(channelNames, channelName), 1);
snrIndex = find(snrValues == snrDb, 1);
assert(numClasses == 15 && signalLength == 512, ...
    'Expected the formal 15-class, 512-sample dataset.');
if isempty(channelIndex) || isempty(snrIndex)
    error('The requested AWGN/18 dB condition is absent from the dataset.');
end
plotRange = 1:128;
fullSamples = zeros(numClasses, 2, signalLength, 'single');
examples = repmat(struct('modulation', '', 'channel', channelName, ...
    'snrDb', snrDb, 'matlabSampleIndex', 0, 'zeroBasedSampleIndex', 0, ...
    'cellFirstMatlabIndex', 0, 'cellLastMatlabIndex', 0, ...
    'offsetWithinCellOneBased', 0, 'classLabel', 0, 'channelLabel', 0, ...
    'splitLabel', 0, 'earlierTestExamplesInCell', 0), numClasses, 1);
for classIndex = 1:numClasses
    cellIndex = ((classIndex - 1) * numChannels + channelIndex - 1) ...
        * numSnrs + snrIndex - 1;
    firstIndex = cellIndex * numPerCell + 1;
    lastIndex = firstIndex + numPerCell - 1;
    cellSplit = m.split(firstIndex:lastIndex, 1);
    firstTestOffset = find(cellSplit == 2, 1, 'first');
    if isempty(firstTestOffset)
        error('No held-out test example for %s.', classNames{classIndex});
    end
    sampleIndex = firstIndex + firstTestOffset - 1;
    actualClass = double(m.y(sampleIndex, 1));
    actualChannel = double(m.channel(sampleIndex, 1));
    actualSnr = double(m.snr(sampleIndex, 1));
    actualSplit = double(m.split(sampleIndex, 1));
    assert(actualClass == classIndex - 1 && actualChannel == channelIndex - 1 ...
        && actualSnr == snrDb && actualSplit == 2, ...
        'Indexed sample does not match its expected stored labels.');
    xi = m.X(sampleIndex, :, :);
    assert(isequal(size(xi), [1, 2, signalLength]) && all(isfinite(xi(:))), ...
        'Unexpected signal dimensions or non-finite stored values.');
    fullSamples(classIndex, :, :) = xi;
    examples(classIndex) = struct('modulation', classNames{classIndex}, ...
        'channel', channelName, 'snrDb', snrDb, ...
        'matlabSampleIndex', sampleIndex, 'zeroBasedSampleIndex', sampleIndex - 1, ...
        'cellFirstMatlabIndex', firstIndex, 'cellLastMatlabIndex', lastIndex, ...
        'offsetWithinCellOneBased', firstTestOffset, 'classLabel', actualClass, ...
        'channelLabel', actualChannel, 'splitLabel', actualSplit, ...
        'earlierTestExamplesInCell', sum(cellSplit(1:firstTestOffset-1) == 2));
end
plottedSamples = fullSamples(:, :, plotRange);
maxAmplitude = double(max(abs(plottedSamples(:))));
amplitudeLimit = max(0.5, ceil(maxAmplitude * 1.06 * 2) / 2);
colors = [68, 119, 170; 68, 170, 153] / 255;
source = dir(datasetPath);
sourceModified = char(datetime(source.datenum, 'ConvertFrom', 'datenum', ...
    'Format', 'yyyy-MM-dd HH:mm:ss'));
provenance = struct('datasetFile', datasetPath, 'sourceFileBytes', source.bytes, ...
    'sourceFileModified', sourceModified, 'generationComplete', true, ...
    'generatedSamples', double(m.generatedSamples), 'datasetSeed', double(meta.seed), ...
    'sampleOrder', meta.sampleOrder, ...
    'selectionRule', 'First stored test example within every requested cell', ...
    'independentRealizations', true, 'signalLength', signalLength, ...
    'plottedSampleRangeOneBased', [plotRange(1), plotRange(end)], ...
    'channel', channelName, 'snrDb', snrDb, ...
    'figurePixels', [1280, 520], 'layoutRowsColumns', [3, 5], ...
    'displayXLimits', [0, 136], ...
    'sharedYLimits', [-amplitudeLimit, amplitudeLimit], ...
    'traceColorsRGB', colors, 'examples', examples);

fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'pixels', ...
    'Position', [60, 60, 1280, 520], 'Renderer', 'painters');
layout = tiledlayout(fig, 3, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
axesHandles = gobjects(numClasses, 1);
traceHandles = gobjects(2, 1);
for classIndex = 1:numClasses
    ax = nexttile(layout);
    axesHandles(classIndex) = ax;
    hold(ax, 'on');
    for componentIndex = 1:2
        trace = plot(ax, plotRange, ...
            reshape(plottedSamples(classIndex, componentIndex, :), 1, []), ...
            'Color', colors(componentIndex, :), 'LineWidth', 0.85);
        if classIndex == 1
            traceHandles(componentIndex) = trace;
        end
    end
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 13, ...
        'XColor', 'k', 'YColor', 'k', 'LineWidth', 0.65, 'Box', 'on', ...
        'TickDir', 'in', 'GridColor', [0, 0, 0], 'GridAlpha', 0.10, ...
        'GridLineStyle', ':', 'Layer', 'bottom');
    % Keep the final tick label inside its panel so adjacent labels do not run together.
    xlim(ax, [0, 136]);
    ylim(ax, [-amplitudeLimit, amplitudeLimit]);
    xticks(ax, [1, 64, 128]);
    yticks(ax, [-amplitudeLimit, 0, amplitudeLimit]);
    grid(ax, 'on');
    if classIndex <= 10
        ax.XTickLabel = [];
    end
    if mod(classIndex - 1, 5) ~= 0
        ax.YTickLabel = [];
    end
    title(ax, classNames{classIndex}, 'FontName', 'Times New Roman', ...
        'FontSize', 17, 'FontWeight', 'bold', 'Color', 'k', 'Interpreter', 'none');
end
xlabel(layout, 'Sample index', 'FontName', 'Times New Roman', ...
    'FontSize', 17, 'Color', 'k');
ylabel(layout, 'Normalized amplitude', 'FontName', 'Times New Roman', ...
    'FontSize', 17, 'Color', 'k');
lgd = legend(axesHandles(1), traceHandles, {'I', 'Q'}, ...
    'FontName', 'Times New Roman', 'FontSize', 17, 'TextColor', 'k', ...
    'Orientation', 'horizontal', 'Box', 'off', 'AutoUpdate', 'off');
lgd.Layout.Tile = 'north';
fig.UserData = struct('provenance', provenance, 'fullSamples', fullSamples, ...
    'plottedSamples', plottedSamples, 'classNames', {classNames}, ...
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
