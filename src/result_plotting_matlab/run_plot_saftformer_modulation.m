%RUN_PLOT_SAFTFORMER_MODULATION Plot all 15 formal Ours SNR-accuracy curves.
% Uses every measured point of the current Full best.pt test evaluation.
clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
evalDir = fullfile(scriptDir, '..', 'runs', 'eval_saftformer_full');
csvFile = fullfile(evalDir, 'modulation_snr_accuracy.csv');
options.ProvenanceFile = fullfile(evalDir, 'evaluation_provenance.json');
options.ClassOrder = {
    'BPSK', 'QPSK', '8PSK', '2FSK', '4FSK', 'CPFSK', 'GFSK', ...
    'PAM4', 'QAM16', 'QAM64', 'QAM256', 'AM-DSB', 'AM-SSB', 'WBFM', 'LFM'
};
options.ExpectedSNR = (-20:2:18)';
options.AccuracyScale = 'percent';
options.Title = '';
options.FigureSize = [960, 800];
options.FontSize = 14;
options.AxisLabelFontSize = 14;
options.LegendFontSize = 14;
options.LegendColumns = 5;
% Natural-size southeast legend uses the empty region below the QAM curves.
% Five columns / three rows and short line samples keep its upper edge low.
options.LegendLocation = 'southeast';
options.LegendItemTokenSize = [16, 10];
options.AxesPosition = [0.11, 0.10, 0.84, 0.84];
options.LineWidth = 1.1;
options.MarkerSize = 3.5;
% Retain the comparison plot's 0.5 dB end margins; also pad the accuracy
% limits so neither near-zero points nor 100% plateaus touch the frame.
options.XLimits = [-20.5, 18.5];
options.YLimits = [-2, 102];
options.YTicks = 0:20:100;
options.GridAlpha = 1;
options.ExportFormats = {'pdf', 'fig'};
outputPath = fullfile(scriptDir, 'figures', 'saftformer_modulation_snr_accuracy.png');
[fig, curveData] = plot_modulation_snr_accuracy(csvFile, outputPath, options);
fprintf('Plotted %d classes x %d measured SNR levels, without smoothing.\n', ...
    numel(curveData.classNames), numel(curveData.snr));
