%RUN_PLOT_SAFTFORMER_MODULATION_RADAR One mean-accuracy curve for Ours.
% All spokes share the same absolute 0--100% radial scale. Each class uses
% all twenty SNR levels and all eleven channel conditions in the formal CSV.
clc;
clear;
close all;

scriptDir = fileparts(mfilename('fullpath'));
evalDir = fullfile(scriptDir, '..', 'runs', 'eval_saftformer_full');
csvFile = fullfile(evalDir, 'modulation_snr_accuracy.csv');
options.ProvenanceFile = fullfile(evalDir, 'evaluation_provenance.json');
options.ConfusionMatrixFile = fullfile(evalDir, 'confusion_matrix.csv');
options.ClassOrder = {
    'BPSK', 'QPSK', '8PSK', '2FSK', '4FSK', 'CPFSK', 'GFSK', ...
    'PAM4', 'QAM16', 'QAM64', 'QAM256', 'AM-DSB', 'AM-SSB', 'WBFM', 'LFM'
};
options.ModelName = 'SAFTFormer (Ours)';
options.FigureSize = [1000, 800];
options.FontSize = 18;
options.RadialFontSize = 14;
options.Color = [238, 102, 119] / 255;
options.LineWidth = 1.5;
options.MarkerSize = 4.5;
outputPath = fullfile(scriptDir, 'figures', 'saftformer_modulation_accuracy_radar.png');
[fig, radarData] = plot_modulation_accuracy_radar(csvFile, outputPath, options);
disp(table(radarData.classNames, radarData.meanAccuracyPercent, ...
    'VariableNames', {'Modulation', 'MeanAccuracyPercent'}));
