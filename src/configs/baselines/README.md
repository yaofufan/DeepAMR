# Baseline models

Nine comparison models are available through the shared training, evaluation,
and benchmark commands. Run commands from `src/`.

| Model | Configuration |
| --- | --- |
| O'Shea ResNet | [resnet.json](resnet.json) |
| MCLDNN | [mcldnn.json](mcldnn.json) |
| CCTL-Net | [cctl_net.json](cctl_net.json) |
| SE-TFA | [se_tfa.json](se_tfa.json) |
| AMC-Net | [amcnet.json](amcnet.json) |
| MCformer | [mcformer.json](mcformer.json) |
| MCCFN | [mccfn.json](mccfn.json) |
| TLDNN | [tldnn.json](tldnn.json) |
| DHN | [dhn.json](dhn.json) |

For example:

```bash
python -m amc.train --config configs/baselines/resnet.json --no-resume
python -m amc.evaluate --config configs/baselines/resnet.json --checkpoint runs/oshea_resnet/best.pt --output-dir runs/eval_oshea_resnet
python -m amc.benchmark --config configs/baselines/resnet.json --checkpoint runs/oshea_resnet/best.pt --metrics runs/eval_oshea_resnet/metrics.json --output runs/eval_oshea_resnet/benchmark.json
```

Choose the matching configuration and checkpoint for each model. The dataset
and checkpoints must be obtained separately or generated through training.
See the [comparison results](../../../README.md#results) for accuracy,
parameter counts, MACs and latency, and the
[framework guide](../../README.md) for the command workflow.
