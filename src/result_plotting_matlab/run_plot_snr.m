%RUN_PLOT_SNR Create a paper-style SNR accuracy figure.

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fullfile(scriptDir, '..', '..');

exp(1).title = 'MATLAB AMR Dataset';
exp(1).metricsFiles = {
    fullfile(projectRoot, 'src', 'runs', 'eval_oshea_resnet', 'snr_accuracy.csv')
};
exp(1).modelNames = {'ResNet'};
exp(1).insetXLim = [0, 18];
exp(1).insetYLim = [0.82, 0.92];

outputPath = fullfile(scriptDir, 'figures', 'snr_accuracy_paper.png');
plot_snr_accuracy_paper(exp, outputPath);
