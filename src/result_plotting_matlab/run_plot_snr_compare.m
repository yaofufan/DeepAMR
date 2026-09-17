%RUN_PLOT_SNR_COMPARE Plot ResNet vs MCLDNN SNR accuracy curves.

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fullfile(scriptDir, '..', '..');

exp(1).title = 'MATLAB AMR Dataset';
exp(1).metricsFiles = {
    fullfile(projectRoot, 'src', 'runs', 'eval_oshea_resnet', 'snr_accuracy.csv')
    fullfile(projectRoot, 'src', 'runs', 'eval_mcldnn', 'snr_accuracy.csv')
};
exp(1).modelNames = {'ResNet', 'MCLDNN'};
exp(1).insetXLim = [0, 18];
exp(1).insetYLim = [0.82, 0.92];

outputPath = fullfile(scriptDir, 'figures', 'snr_accuracy_compare.png');
plot_snr_accuracy_paper(exp, outputPath);
