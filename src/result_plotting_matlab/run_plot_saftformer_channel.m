%RUN_PLOT_SAFTFORMER_CHANNEL Plot all 11 formal Ours channel-SNR curves.
% Each measured channel/SNR accuracy averages all fifteen modulation types.
clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
evalDir = fullfile(scriptDir, '..', 'runs', 'eval_saftformer_full');
csvFile = fullfile(evalDir, 'channel_snr_accuracy.csv');
options.ProvenanceFile = fullfile(evalDir, 'evaluation_provenance.json');
options.ChannelOrder = {
    'AWGN', 'Rayleigh', 'Rician', 'Nakagami', 'DoublySelective', ...
    'FrequencySelectiveRayleigh', 'FrequencySelectiveRician', ...
    'LognormalShadowing', '3GPP_TDL_A', '3GPP_TDL_C', '3GPP_TDL_E'
};
options.ChannelLabels = {
    'AWGN', 'Rayleigh', 'Rician', 'Nakagami', 'Doubly selective', ...
    'Frequency-selective Rayleigh', 'Frequency-selective Rician', ...
    'Lognormal shadowing', '3GPP TDL-A', '3GPP TDL-C', '3GPP TDL-E'
};
options.FigureSize = [960, 800];
options.AxesPosition = [0.11, 0.10, 0.84, 0.84];
options.FontSize = 14;
options.AxisLabelFontSize = 14;
options.LegendFontSize = 12;
options.LegendColumns = 2;
options.LegendLocation = 'southeast';
options.LegendItemTokenSize = [16, 10];
options.LineWidth = 1.1;
options.MarkerSize = 3.5;
options.GridAlpha = 1;
options.XLimits = [-20.5, 18.5];
options.YLimits = [-2, 102];
outputPath = fullfile(scriptDir, 'figures', 'saftformer_channel_snr_accuracy.png');
[fig, channelCurveData] = plot_channel_snr_accuracy(csvFile, outputPath, options);
fprintf('Plotted %d channels x %d measured SNR levels, without smoothing.\n', ...
    numel(channelCurveData.channelNames), numel(channelCurveData.snr));
