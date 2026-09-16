function generate_amr_dataset(cfg)
%GENERATE_AMR_DATASET Generate an AMR I/Q dataset and export a .mat file.
%
% The project brief asks for common 5G/B5G modulation families over AWGN and
% severe fading channels. This generator creates a balanced dataset over:
%   modulations: PSK, FSK, LFM, and several extended baselines
%   channels   : AWGN, Rayleigh, Rician, Nakagami, DoublySelective,
%                frequency-selective fading, lognormal shadowing, 3GPP TDL
%   splits     : train/val/test = 6:2:2 within each modulation-channel-SNR cell
%
% Output contract for PyTorch:
%   X          single [N, 2, L]
%   y          int64  [N, 1], zero-based modulation labels
%   snr        int64  [N, 1], SNR in dB
%   channel    int64  [N, 1], zero-based channel labels
%   split      int8   [N, 1], 0=train, 1=val, 2=test
%   classNames cell
%   channelNames cell
%   splitNames cell

if nargin < 1
    cfg = amr_dataset_config();
end

cfg = normalize_config(cfg);
validate_config(cfg);
rng(cfg.seed);

classNames = cfg.classNames;
channelNames = cfg.channelNames;
splitNames = cfg.splitNames;
snrs = cfg.snrs(:).';

numClasses = numel(classNames);
numChannels = numel(channelNames);
numSnrs = numel(snrs);
numPerCell = cfg.samplesPerClassPerSnrPerChannel;
numSamples = numClasses * numChannels * numSnrs * numPerCell;
L = cfg.signalLength;

meta = cfg;
meta.numSamples = numSamples;
meta.numClasses = numClasses;
meta.numChannels = numChannels;
meta.numSnrs = numSnrs;
meta.format = 'X [N,2,L], y/channel zero-based int64, split 0=train 1=val 2=test';
meta.sampleOrder = 'class -> channel -> SNR -> sample';
meta.createdAt = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss Z'));

channelRuntimes = initialize_channel_runtimes(channelNames, cfg);
runtimeCleanup = onCleanup(@() release_channel_runtimes(channelRuntimes));
writer = initialize_output_file(cfg, classNames, channelNames, splitNames, meta, ...
    numSamples, L);

cellSplit = make_cell_split(numPerCell, cfg.splitFractions);
sampleIdx = 1;
startTime = tic;
for classIdx = 1:numClasses
    modName = classNames{classIdx};
    for channelIdx = 1:numChannels
        channelName = channelNames{channelIdx};
        for snrIdx = 1:numSnrs
            snrDb = snrs(snrIdx);
            localSplit = cellSplit(randperm(numPerCell));
            cellX = zeros(numPerCell, 2, L, 'single');
            for n = 1:numPerCell
                x = generate_clean_signal(modName, cfg);
                x = apply_channel_model(x, channelName, cfg, ...
                    channelRuntimes{channelIdx});
                x = add_awgn(x, snrDb);
                if cfg.normalizeRms
                    x = normalize_rms(x);
                end

                cellX(n, 1, :) = single(real(x));
                cellX(n, 2, :) = single(imag(x));
            end

            cellIndices = sampleIdx:(sampleIdx + numPerCell - 1);
            writer.X(cellIndices, :, :) = cellX;
            writer.y(cellIndices, 1) = repmat(int64(classIdx - 1), numPerCell, 1);
            writer.snr(cellIndices, 1) = repmat(int64(snrDb), numPerCell, 1);
            writer.channel(cellIndices, 1) = repmat(int64(channelIdx - 1), numPerCell, 1);
            writer.split(cellIndices, 1) = int8(localSplit(:));
            sampleIdx = sampleIdx + numPerCell;
            writer.generatedSamples = int64(sampleIdx - 1);
        end
        fprintf(['Generated %-8s over %-30s class %d/%d, channel %d/%d, ' ...
            'samples %d/%d, elapsed %.1f min\n'], ...
            modName, channelName, classIdx, numClasses, channelIdx, numChannels, ...
            sampleIdx - 1, numSamples, toc(startTime) / 60);
    end
end

writer.generationComplete = true;
clear runtimeCleanup;
validate_output_file(cfg.outputPath, numSamples, L);
fprintf('Completed dataset: %s (%.1f min)\n', cfg.outputPath, toc(startTime) / 60);
end

function writer = initialize_output_file(cfg, classNames, channelNames, splitNames, ...
    meta, numSamples, signalLength)
outputDir = fileparts(cfg.outputPath);
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
if exist(cfg.outputPath, 'file')
    if ~cfg.overwriteExisting
        error(['Dataset already exists: %s\nSet cfg.overwriteExisting=true ' ...
            'only when the existing file may be replaced.'], cfg.outputPath);
    end
    fprintf('Removing existing dataset: %s\n', cfg.outputPath);
    delete(cfg.outputPath);
end

generationComplete = false;
generatedSamples = int64(0);
save(cfg.outputPath, 'classNames', 'channelNames', 'splitNames', 'meta', ...
    'generationComplete', 'generatedSamples', cfg.saveVersion);

writer = matfile(cfg.outputPath, 'Writable', true);
writer.X(numSamples, 2, signalLength) = single(0);
writer.y(numSamples, 1) = int64(0);
writer.snr(numSamples, 1) = int64(0);
writer.channel(numSamples, 1) = int64(0);
writer.split(numSamples, 1) = int8(0);

fprintf('Initialized disk-backed dataset for %d samples.\n', numSamples);
end

function runtimes = initialize_channel_runtimes(channelNames, cfg)
runtimes = cell(1, numel(channelNames));
for idx = 1:numel(channelNames)
    key = channel_key(channelNames{idx});
    switch key
        case {'3GPP_TDL_A', 'TDL_A'}
            runtimes{idx} = create_tdl_runtime('TDL-A', cfg);
        case {'3GPP_TDL_C', 'TDL_C'}
            runtimes{idx} = create_tdl_runtime('TDL-C', cfg);
        case {'3GPP_TDL_E', 'TDL_E'}
            runtimes{idx} = create_tdl_runtime('TDL-E', cfg);
        otherwise
            runtimes{idx} = [];
    end
end
end

function tdl = create_tdl_runtime(profile, cfg)
if ~has_5g_tdl()
    tdl = [];
    return;
end

tdl = nrTDLChannel;
tdl.DelayProfile = profile;
tdl.DelaySpread = cfg.tdlDelaySpreadSeconds;
tdl.MaximumDopplerShift = cfg.tdlMaximumDopplerHz;
tdl.SampleRate = cfg.sampleRate;
tdl.NumTransmitAntennas = 1;
tdl.NumReceiveAntennas = 1;
if isprop(tdl, 'RandomStream')
    tdl.RandomStream = 'Global stream';
end
if isprop(tdl, 'NormalizePathGains')
    tdl.NormalizePathGains = true;
end
if isprop(tdl, 'NormalizeChannelOutputs')
    tdl.NormalizeChannelOutputs = true;
end
end

function release_channel_runtimes(runtimes)
for idx = 1:numel(runtimes)
    runtime = runtimes{idx};
    if ~isempty(runtime) && isLocked(runtime)
        release(runtime);
    end
end
end

function validate_output_file(path, numSamples, signalLength)
m = matfile(path);
assert(isequal(size(m, 'X'), [numSamples, 2, signalLength]), ...
    'Unexpected X shape in generated dataset.');
assert(isequal(size(m, 'y'), [numSamples, 1]), ...
    'Unexpected y shape in generated dataset.');
assert(double(m.generatedSamples) == numSamples, ...
    'Generated-sample count does not match the expected total.');
assert(logical(m.generationComplete), ...
    'Dataset completion flag was not written.');
firstSample = m.X(1, :, :);
lastSample = m.X(numSamples, :, :);
assert(all(isfinite(firstSample(:))) && all(isfinite(lastSample(:))), ...
    'Generated dataset contains non-finite boundary samples.');
end

function cfg = normalize_config(cfg)
if ~isfield(cfg, 'samplesPerClassPerSnrPerChannel')
    cfg.samplesPerClassPerSnrPerChannel = cfg.samplesPerClassPerSnr;
end
if ~isfield(cfg, 'overwriteExisting')
    cfg.overwriteExisting = false;
end
if ~isfield(cfg, 'channelNames')
    cfg.channelNames = { ...
        'AWGN', 'Rayleigh', 'Rician', 'Nakagami', 'DoublySelective', ...
        'FrequencySelectiveRayleigh', 'FrequencySelectiveRician', ...
        'LognormalShadowing', '3GPP_TDL_A', '3GPP_TDL_C', '3GPP_TDL_E' ...
    };
end
if ~isfield(cfg, 'splitNames')
    cfg.splitNames = {'train', 'val', 'test'};
end
if ~isfield(cfg, 'splitFractions')
    cfg.splitFractions = [0.6, 0.2, 0.2];
end
if ~isfield(cfg, 'fskNormalizedToneRange')
    cfg.fskNormalizedToneRange = [-0.35, 0.35];
end
if ~isfield(cfg, 'gfskGaussianWidthSymbols')
    cfg.gfskGaussianWidthSymbols = 0.6;
end
if ~isfield(cfg, 'lfmCenterFrequencyRange')
    cfg.lfmCenterFrequencyRange = [-0.05, 0.05];
end
if ~isfield(cfg, 'lfmSweepBandwidthRange')
    cfg.lfmSweepBandwidthRange = [0.50, 0.70];
end
if ~isfield(cfg, 'wbfmDeviation')
    cfg.wbfmDeviation = 4.0;
end
if ~isfield(cfg, 'frequencySelectiveNumTaps')
    cfg.frequencySelectiveNumTaps = 6;
end
if ~isfield(cfg, 'frequencySelectiveMaxDelay')
    cfg.frequencySelectiveMaxDelay = 10;
end
if ~isfield(cfg, 'frequencySelectiveDecay')
    cfg.frequencySelectiveDecay = 2.0;
end
if ~isfield(cfg, 'lognormalShadowingSigmaDb')
    cfg.lognormalShadowingSigmaDb = 4;
end
if ~isfield(cfg, 'lognormalShadowingCorrelationLength')
    cfg.lognormalShadowingCorrelationLength = 24;
end
if ~isfield(cfg, 'sampleRate')
    cfg.sampleRate = 1e6;
end
if ~isfield(cfg, 'tdlDelaySpreadSeconds')
    cfg.tdlDelaySpreadSeconds = 1e-6;
end
if ~isfield(cfg, 'tdlMaximumDopplerHz')
    cfg.tdlMaximumDopplerHz = 2000;
end
if ~isfield(cfg, 'require5GToolbox')
    cfg.require5GToolbox = false;
end
cfg.splitFractions = cfg.splitFractions / sum(cfg.splitFractions);
end

function validate_config(cfg)
supportedClasses = { ...
    'BPSK', 'QPSK', '8PSK', ...
    '2FSK', '4FSK', 'CPFSK', 'GFSK', 'PAM4', ...
    'QAM16', 'QAM64', 'QAM256', ...
    'AM-DSB', 'AM-SSB', 'WBFM', 'LFM' ...
};
supportedChannels = { ...
    'AWGN', 'RAYLEIGH', 'RICIAN', 'NAKAGAMI', 'DOUBLYSELECTIVE', ...
    'FREQUENCYSELECTIVERAYLEIGH', 'FREQUENCYSELECTIVERICIAN', ...
    'LOGNORMALSHADOWING', '3GPP_TDL_A', '3GPP_TDL_C', '3GPP_TDL_E', ...
    'TDL_A', 'TDL_C', 'TDL_E' ...
};

classKeys = cellfun(@upper, cfg.classNames, 'UniformOutput', false);
channelKeys = cellfun(@channel_key, cfg.channelNames, 'UniformOutput', false);
assert(~isempty(classKeys) && numel(unique(classKeys)) == numel(classKeys), ...
    'classNames must be nonempty and unique.');
assert(all(ismember(classKeys, supportedClasses)), ...
    'classNames contains an unsupported modulation.');
assert(~isempty(channelKeys) && numel(unique(channelKeys)) == numel(channelKeys), ...
    'channelNames must be nonempty and unique.');
assert(all(ismember(channelKeys, supportedChannels)), ...
    'channelNames contains an unsupported channel.');
assert(isnumeric(cfg.snrs) && ~isempty(cfg.snrs) && all(isfinite(cfg.snrs)), ...
    'snrs must contain finite numeric values.');
assert(all(cfg.snrs == round(cfg.snrs)), ...
    'snrs must be integers because the output stores them as int64.');
assert(isscalar(cfg.samplesPerClassPerSnrPerChannel) && ...
    cfg.samplesPerClassPerSnrPerChannel >= 1 && ...
    cfg.samplesPerClassPerSnrPerChannel == round(cfg.samplesPerClassPerSnrPerChannel), ...
    'samplesPerClassPerSnrPerChannel must be a positive integer.');
assert(isscalar(cfg.signalLength) && cfg.signalLength >= 2 && ...
    cfg.signalLength == round(cfg.signalLength), ...
    'signalLength must be an integer greater than one.');
assert(isscalar(cfg.samplesPerSymbol) && cfg.samplesPerSymbol >= 2 && ...
    cfg.samplesPerSymbol == round(cfg.samplesPerSymbol), ...
    'samplesPerSymbol must be an integer greater than one.');
assert(numel(cfg.splitFractions) == 3 && all(cfg.splitFractions >= 0) && ...
    sum(cfg.splitFractions) > 0, ...
    'splitFractions must contain three nonnegative values with a positive sum.');
assert(numel(cfg.fskNormalizedToneRange) == 2 && ...
    cfg.fskNormalizedToneRange(1) < cfg.fskNormalizedToneRange(2), ...
    'fskNormalizedToneRange must be [minimum maximum].');
assert(max(abs(cfg.fskNormalizedToneRange)) < 0.5, ...
    'FSK tones must remain inside the normalized Nyquist interval (-0.5, 0.5).');
assert(numel(cfg.lfmCenterFrequencyRange) == 2 && ...
    cfg.lfmCenterFrequencyRange(1) <= cfg.lfmCenterFrequencyRange(2), ...
    'lfmCenterFrequencyRange must be [minimum maximum].');
assert(numel(cfg.lfmSweepBandwidthRange) == 2 && ...
    cfg.lfmSweepBandwidthRange(1) > 0 && ...
    cfg.lfmSweepBandwidthRange(1) <= cfg.lfmSweepBandwidthRange(2), ...
    'lfmSweepBandwidthRange must contain positive ordered limits.');
maxLfmFrequency = max(abs(cfg.lfmCenterFrequencyRange)) + ...
    cfg.lfmSweepBandwidthRange(2) / 2;
assert(maxLfmFrequency < 0.5, ...
    'The configured LFM sweep can cross the normalized Nyquist limit.');
assert(strcmp(cfg.saveVersion, '-v7.3'), ...
    'Disk-backed generation requires cfg.saveVersion=''-v7.3''.');

usesTdl = any(ismember(channelKeys, ...
    {'3GPP_TDL_A', '3GPP_TDL_C', '3GPP_TDL_E', 'TDL_A', 'TDL_C', 'TDL_E'}));
if usesTdl && cfg.require5GToolbox && ~has_5g_tdl()
    error(['The formal configuration requires 5G Toolbox nrTDLChannel. ' ...
        'Install/enable 5G Toolbox or explicitly set cfg.require5GToolbox=false ' ...
        'to use the nonstandard fallback for a pipeline test.']);
end
end

function available = has_5g_tdl()
available = exist('nrTDLChannel', 'file') == 2 || ...
    exist('nrTDLChannel', 'class') == 8;
end

function key = channel_key(channelName)
key = upper(strrep(strtrim(channelName), '-', '_'));
end

function split = make_cell_split(n, fractions)
nTrain = round(n * fractions(1));
nVal = round(n * fractions(2));
nTest = n - nTrain - nVal;
if nTest < 0
    nTest = 0;
    nVal = max(0, n - nTrain);
end
split = [zeros(1, nTrain), ones(1, nVal), 2 * ones(1, nTest)];
if numel(split) < n
    split = [split, 2 * ones(1, n - numel(split))];
end
split = split(1:n);
end

function x = generate_clean_signal(modName, cfg)
switch upper(modName)
    case 'BPSK'
        x = generate_psk(2, cfg);
    case 'QPSK'
        x = generate_psk(4, cfg);
    case '8PSK'
        x = generate_psk(8, cfg);
    case '2FSK'
        x = generate_mfsk(2, cfg, false, false);
    case '4FSK'
        x = generate_mfsk(4, cfg, false, false);
    case 'CPFSK'
        x = generate_mfsk(2, cfg, true, false);
    case 'GFSK'
        x = generate_mfsk(2, cfg, true, true);
    case 'LFM'
        x = generate_lfm(cfg);
    case 'PAM4'
        x = generate_pam4(cfg);
    case 'QAM16'
        x = generate_square_qam(16, cfg);
    case 'QAM64'
        x = generate_square_qam(64, cfg);
    case 'QAM256'
        x = generate_square_qam(256, cfg);
    case 'AM-DSB'
        x = generate_am_dsb(cfg);
    case 'AM-SSB'
        x = generate_am_ssb(cfg);
    case 'WBFM'
        x = generate_wbfm(cfg);
    otherwise
        error('Unsupported modulation type: %s', modName);
end
end

function x = generate_psk(order, cfg)
numSymbols = symbols_needed(cfg);
idx = randi([0, order - 1], 1, numSymbols);
symbols = exp(1j * (2 * pi * idx / order + pi / order));
x = pulse_shape(symbols, cfg);
end

function x = generate_pam4(cfg)
numSymbols = symbols_needed(cfg);
levels = [-3, -1, 1, 3] / sqrt(5);
symbols = levels(randi([1, 4], 1, numSymbols));
x = pulse_shape(symbols, cfg);
end

function x = generate_square_qam(order, cfg)
side = round(sqrt(order));
levels = -(side - 1):2:(side - 1);
numSymbols = symbols_needed(cfg);
i = levels(randi([1, side], 1, numSymbols));
q = levels(randi([1, side], 1, numSymbols));
symbols = complex(i, q);
symbols = symbols / sqrt(mean(abs(symbols).^2));
x = pulse_shape(symbols, cfg);
end

function x = generate_mfsk(order, cfg, continuousPhase, useGaussian)
L = cfg.signalLength;
sps = cfg.samplesPerSymbol;
numSymbols = symbols_needed(cfg);
toneIdx = randi([0, order - 1], 1, numSymbols);
tones = linspace(cfg.fskNormalizedToneRange(1), ...
    cfg.fskNormalizedToneRange(2), order);
freq = repelem(tones(toneIdx + 1), sps);
freq = fit_length(freq, L);
if useGaussian
    width = max(3, round(cfg.gfskGaussianWidthSymbols * sps));
    freq = smooth_gaussian(freq, width);
end
if continuousPhase
    phase = cumsum(2 * pi * freq);
else
    sampleInSymbol = repmat(0:sps - 1, 1, numSymbols);
    sampleInSymbol = fit_length(sampleInSymbol, L);
    phase = 2 * pi * freq .* sampleInSymbol;
end
x = exp(1j * phase);
end

function x = generate_lfm(cfg)
L = cfg.signalLength;
sampleIndex = 0:L - 1;
centeredIndex = sampleIndex - (L - 1) / 2;
centerFreq = random_uniform(cfg.lfmCenterFrequencyRange);
sweepBandwidth = random_uniform(cfg.lfmSweepBandwidthRange);
if rand() < 0.5
    sweepBandwidth = -sweepBandwidth;
end
phase = 2 * pi * (centerFreq * sampleIndex + ...
    0.5 * sweepBandwidth * centeredIndex .^ 2 / L);
window = 0.54 - 0.46 * cos(2 * pi * (0:L - 1) / max(1, L - 1));
x = window .* exp(1j * phase);
end

function x = generate_am_dsb(cfg)
L = cfg.signalLength;
t = (0:L - 1) / L;
message = analog_message(t);
x = complex(message, zeros(size(message)));
end

function x = generate_am_ssb(cfg)
L = cfg.signalLength;
t = (0:L - 1) / L;
message = analog_message(t);
x = analytic_signal(message);
end

function x = generate_wbfm(cfg)
L = cfg.signalLength;
t = (0:L - 1) / L;
message = analog_message(t);
message = message / max(abs(message) + eps);
deviation = cfg.wbfmDeviation;
phase = 2 * pi * deviation * cumsum(message) / L;
x = exp(1j * phase);
end

function message = analog_message(t)
f1 = 1 + 5 * rand();
f2 = 6 + 8 * rand();
message = 0.65 * sin(2 * pi * f1 * t + 2 * pi * rand()) + ...
          0.35 * sin(2 * pi * f2 * t + 2 * pi * rand());
message = message + 0.05 * randn(size(message));
message = message / max(abs(message) + eps);
end

function x = pulse_shape(symbols, cfg)
sps = cfg.samplesPerSymbol;
if cfg.useRrcIfAvailable && exist('rcosdesign', 'file') == 2
    up = zeros(1, numel(symbols) * sps);
    up(1:sps:end) = symbols;
    h = rcosdesign(cfg.rrcRollOff, cfg.rrcSpan, sps, 'sqrt');
    x = conv(up, h, 'same');
else
    x = repelem(symbols, sps);
end
x = fit_length(x, cfg.signalLength);
end

function x = apply_channel_model(x, channelName, cfg, channelRuntime)
L = cfg.signalLength;
n = 0:L - 1;
channelKey = channel_key(channelName);

switch channelKey
    case 'AWGN'
        h = 1;
    case 'RAYLEIGH'
        h = rayleigh_taps(cfg);
    case 'RICIAN'
        h = rician_taps(cfg);
    case 'NAKAGAMI'
        h = nakagami_taps(cfg);
    case 'DOUBLYSELECTIVE'
        h = rayleigh_taps(cfg);
        x = filter(h, 1, x);
        x = apply_time_selective_fading(x, cfg);
        h = 1;
    case 'FREQUENCYSELECTIVERAYLEIGH'
        h = frequency_selective_rayleigh_taps(cfg);
    case 'FREQUENCYSELECTIVERICIAN'
        h = frequency_selective_rician_taps(cfg);
    case 'LOGNORMALSHADOWING'
        x = apply_lognormal_shadowing(x, cfg);
        h = 1;
    case {'3GPP_TDL_A', 'TDL_A'}
        x = apply_3gpp_tdl_channel(x, 'TDL-A', cfg, channelRuntime);
        h = 1;
    case {'3GPP_TDL_C', 'TDL_C'}
        x = apply_3gpp_tdl_channel(x, 'TDL-C', cfg, channelRuntime);
        h = 1;
    case {'3GPP_TDL_E', 'TDL_E'}
        x = apply_3gpp_tdl_channel(x, 'TDL-E', cfg, channelRuntime);
        h = 1;
    otherwise
        error('Unsupported channel type: %s', channelName);
end

x = filter(h, 1, x);

if cfg.enableCfo
    cfo = (2 * rand() - 1) * cfg.maxCfoCyclesPerSample;
    x = x .* exp(1j * 2 * pi * cfo * n);
end

if cfg.enableRandomPhase
    x = x .* exp(1j * 2 * pi * rand());
end
end

function h = rayleigh_taps(cfg)
delays = 0:cfg.maxMultipathDelay;
envelope = exp(-delays / max(cfg.multipathDecay, eps));
h = (randn(size(delays)) + 1j * randn(size(delays))) .* envelope;
h = h / sqrt(sum(abs(h).^2) + eps);
end

function h = rician_taps(cfg)
ray = rayleigh_taps(cfg);
k = cfg.ricianKFactor;
los = zeros(size(ray));
los(1) = 1;
h = sqrt(k / (k + 1)) * los + sqrt(1 / (k + 1)) * ray;
h = h / sqrt(sum(abs(h).^2) + eps);
end

function h = frequency_selective_rayleigh_taps(cfg)
maxDelay = max(1, round(cfg.frequencySelectiveMaxDelay));
numTaps = min(maxDelay + 1, max(2, round(cfg.frequencySelectiveNumTaps)));
if numTaps > 1
    delays = sort([0, randperm(maxDelay, numTaps - 1)]);
else
    delays = 0;
end
power = exp(-delays / max(cfg.frequencySelectiveDecay, eps));
coeffs = (randn(size(delays)) + 1j * randn(size(delays))) .* sqrt(power / 2);
h = zeros(1, maxDelay + 1);
h(delays + 1) = coeffs;
h = h / sqrt(sum(abs(h).^2) + eps);
end

function h = frequency_selective_rician_taps(cfg)
ray = frequency_selective_rayleigh_taps(cfg);
k = cfg.ricianKFactor;
los = zeros(size(ray));
los(1) = 1;
h = sqrt(k / (k + 1)) * los + sqrt(1 / (k + 1)) * ray;
h = h / sqrt(sum(abs(h).^2) + eps);
end

function y = apply_lognormal_shadowing(x, cfg)
L = numel(x);
width = max(3, round(cfg.lognormalShadowingCorrelationLength));
shadow = smooth_gaussian(randn(1, L), width);
shadow = shadow - mean(shadow);
shadow = shadow / (std(shadow) + eps);
shadowDb = cfg.lognormalShadowingSigmaDb * shadow;
amplitude = 10 .^ (shadowDb / 20);
y = x .* amplitude;
end

function y = apply_3gpp_tdl_channel(x, profile, cfg, tdl)
if ~isempty(tdl)
    if isLocked(tdl)
        reset(tdl);
    end
    rx = tdl(x(:));
    y = rx(:, 1).';
else
    h = approximate_tdl_taps(profile, cfg);
    y = filter(h, 1, x);
    y = apply_time_selective_fading(y, cfg);
end
y = fit_length(y, numel(x));
end

function value = random_uniform(limits)
value = limits(1) + (limits(2) - limits(1)) * rand();
end

function h = approximate_tdl_taps(profile, cfg)
switch upper(profile)
    case 'TDL-A'
        delays = [0, 0.3819, 0.5868, 1.5375, 2.2242, 3.0582, 4.0810, 5.3043, 9.6586];
        gainsDb = [-13.4, 0.0, -4.0, -15.9, -16.7, -11.3, -12.7, -19.9, -29.7];
        losK = 0;
    case 'TDL-C'
        delays = [0, 0.2099, 0.2329, 0.6366, 1.6374, 2.2948, 3.4744, 5.4924, 8.3427];
        gainsDb = [-4.4, -1.2, -5.2, 0.0, -7.4, -10.7, -11.1, -15.7, -22.8];
        losK = 0;
    case 'TDL-E'
        delays = [0, 0.5133, 0.5440, 0.5630, 0.5440, 0.7112, 1.9092, 5.0067];
        gainsDb = [-0.03, -22.0, -15.8, -18.1, -19.8, -22.9, -22.4, -24.6];
        losK = cfg.ricianKFactor;
    otherwise
        error('Unsupported TDL profile: %s', profile);
end
sampleDelays = round(delays * cfg.tdlDelaySpreadSeconds * cfg.sampleRate);
maxDelay = max(sampleDelays);
power = 10 .^ (gainsDb / 10);
coeffs = (randn(size(power)) + 1j * randn(size(power))) .* sqrt(power / 2);
if losK > 0
    coeffs(1) = sqrt(losK / (losK + 1)) + sqrt(1 / (losK + 1)) * coeffs(1);
end
h = zeros(1, maxDelay + 1);
for idx = 1:numel(sampleDelays)
    h(sampleDelays(idx) + 1) = h(sampleDelays(idx) + 1) + coeffs(idx);
end
h = h / sqrt(sum(abs(h).^2) + eps);
end

function h = nakagami_taps(cfg)
delays = 0:cfg.maxMultipathDelay;
envelope = exp(-delays / max(cfg.multipathDecay, eps));
m = cfg.nakagamiM;
power = gamma_unit_mean(m, size(delays));
phase = 2 * pi * rand(size(delays));
h = sqrt(power) .* exp(1j * phase) .* envelope;
h = h / sqrt(sum(abs(h).^2) + eps);
end

function power = gamma_unit_mean(shape, outSize)
% Base MATLAB replacement for gamrnd(shape, 1 / shape, outSize).
shapeInt = max(1, round(shape));
power = zeros(outSize);
for k = 1:shapeInt
    power = power - log(max(rand(outSize), eps));
end
power = power / shapeInt;
end

function y = apply_time_selective_fading(x, cfg)
L = numel(x);
n = 0:L - 1;
fd = (2 * rand() - 1) * cfg.maxDopplerCyclesPerSample;
slowPhase = exp(1j * 2 * pi * fd * n);
segLen = max(4, cfg.doublySelectiveSegmentLength);
numSeg = ceil(L / segLen);
g = zeros(1, L);
for s = 1:numSeg
    idxStart = (s - 1) * segLen + 1;
    idxEnd = min(L, s * segLen);
    coeff = (randn() + 1j * randn()) / sqrt(2);
    g(idxStart:idxEnd) = coeff;
end
g = smooth_complex(g, segLen);
g = g / sqrt(mean(abs(g).^2) + eps);
y = x .* g .* slowPhase;
end

function y = add_awgn(x, snrDb)
signalPower = mean(abs(x).^2);
noisePower = signalPower / (10 ^ (snrDb / 10));
noise = sqrt(noisePower / 2) * (randn(size(x)) + 1j * randn(size(x)));
y = x + noise;
end

function x = normalize_rms(x)
x = x / sqrt(mean(abs(x).^2) + eps);
end

function n = symbols_needed(cfg)
n = ceil(cfg.signalLength / cfg.samplesPerSymbol) + cfg.rrcSpan + 2;
end

function y = fit_length(x, L)
if numel(x) < L
    x = [x, repmat(x(end), 1, L - numel(x))];
end
y = x(1:L);
end

function y = smooth_gaussian(x, width)
n = -width:width;
sigma = max(width / 2, eps);
g = exp(-(n .^ 2) / (2 * sigma ^ 2));
g = g / sum(g);
y = conv(x, g, 'same');
end

function y = smooth_complex(x, width)
y = smooth_gaussian(real(x), width) + 1j * smooth_gaussian(imag(x), width);
end

function z = analytic_signal(x)
if exist('hilbert', 'file') == 2
    z = hilbert(x);
    return;
end

N = numel(x);
Xf = fft(x);
h = zeros(1, N);
if mod(N, 2) == 0
    h(1) = 1;
    h(N / 2 + 1) = 1;
    h(2:N / 2) = 2;
else
    h(1) = 1;
    h(2:(N + 1) / 2) = 2;
end
z = ifft(Xf .* h);
end
