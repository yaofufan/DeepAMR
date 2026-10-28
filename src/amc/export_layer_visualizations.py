"""Extract ICSA gates, STF-GLU conditions and layer features from SAFTFormer.

Run from src: python -m amc.export_layer_visualizations
Uses the formal best.pt on a deterministic, condition-balanced test subset.
Hooks observe the original FP32/eval forward without replacing any operation.
Exports MATLAB data and a JSON audit. No training or evaluation file is changed.

Relative response strength is condition RMS / pre-condition branch RMS.
It measures activation magnitude, not prediction attribution or accuracy gain.
The two temporal calls are the actual A/B branch calls, not four-scale routing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader

from amc.config import load_config, resolve_src_path
from amc.data.datamodule import H5MatSignalDataset
from amc.data.matlab import load_matlab_hdf5_metadata
from amc.models import build_model
from amc.training.checkpoint import load_checkpoint
from amc.training.engine import resolve_device


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LayerCapture:
    """Observe native outputs. Hooks are removed on exit, including failures."""

    def __init__(self, model, include_hidden: bool = False) -> None:
        self.model = model
        self.include_hidden = include_hidden
        self.handles = []
        self.clear()

    def clear(self) -> None:
        self.scores = []
        self.spectral = []
        self.features = []
        self.temporal = [[] for _ in self.model.blocks]

    def __enter__(self):
        embed = self.model.patch_embed
        if not hasattr(embed, "symbol_score") or self.model.spectral_conditioner is None:
            raise ValueError("This exporter requires the Full model with ICSA and spectral conditioning.")
        if any(getattr(block.ffn, "temporal", None) is None for block in self.model.blocks):
            raise ValueError("This exporter requires temporal conditioning in every block.")
        self.handles.append(embed.symbol_score.register_forward_hook(
            lambda _m, _i, output: self.scores.append(output.detach().cpu())))
        self.handles.append(embed.register_forward_hook(
            lambda _m, _i, output: self.features.append(output.detach().cpu())))
        self.handles.append(self.model.spectral_conditioner.register_forward_hook(
            lambda _m, _i, output: self.spectral.append(output.detach().cpu())))
        for index, block in enumerate(self.model.blocks):
            def temporal_hook(_module, inputs, output, index=index):
                # Native input/output shape [batch,192,59].
                self.temporal[index].append((inputs[0].detach().cpu(), output.detach().cpu()))

            self.handles.append(block.ffn.temporal.register_forward_hook(temporal_hook))
            self.handles.append(block.register_forward_hook(
                lambda _m, _i, output: self.features.append(
                    output[:, self.model.prefix_tokens:].detach().cpu())))
        return self

    def __exit__(self, *_args) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def collect(self) -> dict[str, np.ndarray]:
        depth = len(self.model.blocks)
        if len(self.scores) != 1 or len(self.spectral) != 1 or len(self.features) != depth + 1:
            raise RuntimeError("Unexpected embedding/conditioner/block hook invocation count.")
        if any(len(calls) != 2 for calls in self.temporal):
            raise RuntimeError("Expected exactly two native temporal calls (A and B) per block.")
        # All reductions use CPU float32 tensors captured from the actual forward.
        scores = self.scores[0].float()
        weights = 1 + self.model.patch_embed.gate_scale * scores.tanh()
        context = self.spectral[0].float()
        raw_a, raw_b, time_a, time_b = [], [], [], []
        for calls in self.temporal:
            raw_a.append(calls[0][0].transpose(1, 2).float())
            raw_b.append(calls[1][0].transpose(1, 2).float())
            time_a.append(calls[0][1].transpose(1, 2).float())
            time_b.append(calls[1][1].transpose(1, 2).float())
        a, b, ta, tb = [torch.stack(parts, dim=1) for parts in (raw_a, raw_b, time_a, time_b)]
        scales = torch.stack([block.ffn.frequency_scale.detach().cpu().float()
                              for block in self.model.blocks])
        frequency = context[:, None] * scales[None, :, None, None]
        branch_rms = ((a.square() + b.square()).mean(-1) / 2).sqrt()
        temporal_rms = ((ta.square() + tb.square()).mean(-1) / 2).sqrt()
        frequency_rms = frequency.square().mean(-1).sqrt()
        denominator = branch_rms.clamp_min(1e-8)
        scale_rms = []
        cursor = 0
        for width in self.model.blocks[0].ffn.temporal.split_sizes:
            group = slice(cursor, cursor + width)
            scale_rms.append(((ta[..., group].square() + tb[..., group].square()).mean(-1) / 2).sqrt())
            cursor += width
        result = {
            "icsa_weights": weights[..., 0],  # [N,59,6], bounded [0.5,1.5]
            "icsa_scores": scores[..., 0],
            "spectral_context": context,  # [N,59,192], before per-layer scale
            "frequency_scales": scales,
            "patch_features": torch.stack(self.features, dim=1),  # [N,7,59,48]
            "branch_rms": branch_rms,
            "temporal_rms": temporal_rms,
            "frequency_rms": frequency_rms,
            "temporal_relative_rms": temporal_rms / denominator,
            "frequency_relative_rms": frequency_rms / denominator,
            "temporal_scale_rms": torch.stack(scale_rms, dim=2),  # [N,6,4,59]
        }
        if self.include_hidden:
            result.update(branch_a=a, branch_b=b, temporal_a=ta, temporal_b=tb)
        return {name: value.numpy().astype(np.float32) for name, value in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ours_saftformer/full.json")
    parser.add_argument("--checkpoint", default="runs/saftformer_full/best.pt")
    parser.add_argument("--evaluation-dir", default="runs/eval_saftformer_full")
    parser.add_argument("--output-dir", default="result_plotting_matlab/feature_data")
    parser.add_argument("--classes", nargs="+", default=["BPSK", "QAM64", "LFM"])
    parser.add_argument("--channels", nargs="+", default=["AWGN", "DoublySelective"])
    parser.add_argument("--snr-levels", nargs="+", type=int, default=[0, 10])
    parser.add_argument("--samples-per-cell", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-hidden", action="store_true",
                        help="Also export raw A/B branch and temporal-condition tensors (larger MAT).")
    args = parser.parse_args()
    if args.samples_per_cell < 1 or args.batch_size < 1:
        parser.error("Sample count and batch size must be positive.")
    for name in ("classes", "channels", "snr_levels"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(f"Duplicate entries in --{name.replace('_','-')}.")
    config_path = resolve_src_path(args.config).resolve()
    checkpoint_path = resolve_src_path(args.checkpoint).resolve()
    evaluation_dir = resolve_src_path(args.evaluation_dir).resolve()
    output_dir = resolve_src_path(args.output_dir).resolve()
    config = load_config(config_path)
    dataset_path = resolve_src_path(config.data.path).resolve()
    provenance_path = evaluation_dir / "evaluation_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != provenance["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the formal evaluation checkpoint.")
    if provenance.get("checkpoint_selection") != "best.pt" or provenance.get("split") != "test":
        raise ValueError("Expected the formal best.pt test evaluation.")
    y, snr, class_names, metadata = load_matlab_hdf5_metadata(
        dataset_path, y_key=config.data.mat_y_key, snr_key=config.data.mat_snr_key,
        channel_key=config.data.mat_channel_key, split_key=config.data.mat_split_key,
        class_names=config.data.class_names, channel_names=config.data.channel_names)
    channel, split = metadata["channel"], metadata["split"]
    if split is None:
        raise ValueError("Stored test split labels are required.")
    channel_names = metadata["channel_names"]
    for requested, available, name in [(args.classes, class_names, "classes"),
                                        (args.channels, channel_names, "channels"),
                                        (args.snr_levels, np.unique(snr).tolist(), "SNR levels")]:
        if any(value not in available for value in requested):
            raise ValueError(f"Unsupported {name}: {requested}.")
    test_indices = np.flatnonzero(split == 2)
    reference = loadmat(evaluation_dir / "predictions.mat", variable_names=["y_true", "y_pred", "snr", "channel"])
    for actual, field in [(y,"y_true"),(snr,"snr"),(channel,"channel")]:
        if not np.array_equal(actual[test_indices], reference[field].ravel()):
            raise ValueError(f"Formal predictions are not in ascending test-source order: {field}.")
    rng = np.random.default_rng(args.seed)
    selected = []
    cells = []
    # Selection uses only class/SNR/channel and seed, never predictions or gates.
    for class_name in args.classes:
        for channel_name in args.channels:
            for level in args.snr_levels:
                candidates = np.flatnonzero((split == 2) & (y == class_names.index(class_name))
                    & (channel == channel_names.index(channel_name)) & (snr == level))
                if len(candidates) < args.samples_per_cell:
                    raise ValueError("Insufficient test examples in a requested condition cell.")
                indices = rng.choice(candidates, args.samples_per_cell, replace=False)
                selected.extend(indices.tolist())
                cells.append(dict(modulation=class_name,channel=channel_name,snr=level,
                                  source_indices=indices.tolist()))
    source_indices = np.asarray(selected,dtype=np.int64)
    assert np.all(split[source_indices] == 2) and len(np.unique(source_indices)) == len(source_indices)
    dataset = H5MatSignalDataset(str(dataset_path),config.data.mat_x_key,y,snr,channel,config.data.normalize)
    batches = [source_indices[start:start+args.batch_size].tolist()
               for start in range(0,len(source_indices),args.batch_size)]
    loader = DataLoader(dataset,batch_sampler=batches,num_workers=0)
    checkpoint = load_checkpoint(str(checkpoint_path),map_location="cpu")
    config.model.num_classes = len(class_names)
    config.model.input_length = config.data.input_length
    model = build_model(config.model)
    model.load_state_dict(checkpoint["model"],strict=True)
    device = resolve_device(args.device)
    model.to(device).eval()
    collected: dict[str,list[np.ndarray]] = {}
    inputs, logits_all = [], []
    native_max_difference = 0.0
    print(f"Extracting {len(source_indices)} test examples on {device}; FP32/eval; no autocast.",flush=True)
    try:
        with torch.inference_mode(), LayerCapture(model,args.include_hidden) as capture:
            for batch_index,batch in enumerate(loader):
                x = batch["x"].to(device=device,dtype=torch.float32)
                capture.clear()
                logits = model(x)
                features = capture.collect()
                if batch_index == 0:
                    # Independently reconstruct actual ICSA gates/windows, including projection.
                    embed = model.patch_embed
                    symbols = embed.activation(embed.symbol_encoder(x))
                    span,step = embed.frame_projection.kernel_size[0],embed.frame_projection.stride[0]
                    windows = symbols.unfold(-1,span,step).permute(0,2,3,1)
                    gates = embed.aggregation_weights(windows)
                    np.testing.assert_allclose(gates[...,0].cpu().numpy(),features["icsa_weights"],rtol=1e-6,atol=1e-6)
                    gated = windows*gates
                    expected = embed.frame_projection(gated.permute(0,1,3,2).reshape(-1,embed.symbol_channels,span))
                    expected = expected.squeeze(-1).reshape(len(x),model.num_patches,-1)
                    np.testing.assert_allclose(expected.cpu().numpy(),features["patch_features"][:,0],rtol=1e-5,atol=1e-6)
                    # The observation hooks must not alter inference.
                    model_output = model(x)
                    native_max_difference = float((model_output-logits).abs().max())
                    if native_max_difference > 1e-6:
                        raise RuntimeError("Repeated eval forward changed logits.")
                for name,value in features.items():
                    if name == "frequency_scales":
                        continue
                    collected.setdefault(name,[]).append(value)
                inputs.append(x.cpu().numpy())
                logits_all.append(logits.cpu().numpy())
                print(f"layer batches {batch_index+1}/{len(batches)}",flush=True)
    finally:
        if dataset._h5 is not None:
            dataset._h5.close()
    # Capture context has exited: compare logits against a hook-free forward.
    with torch.inference_mode():
        unobserved = model(torch.from_numpy(inputs[0]).to(device)).cpu().numpy()
    hook_free_difference = float(np.max(np.abs(unobserved-logits_all[0])))
    if hook_free_difference > 1e-6:
        raise RuntimeError("Captured and hook-free model logits differ.")
    arrays = {name:np.concatenate(values,axis=0) for name,values in collected.items()}
    arrays["frequency_scales"] = features["frequency_scales"]
    logits = np.concatenate(logits_all)
    predictions = logits.argmax(1)
    positions = np.searchsorted(test_indices,source_indices)
    mismatches = int(np.count_nonzero(predictions != reference["y_pred"].ravel()[positions]))
    if mismatches:
        raise RuntimeError(f"{mismatches} extracted predictions differ from the formal evaluation.")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("A captured layer contains non-finite values.")
    assert arrays["icsa_weights"].shape == (len(source_indices),59,6)
    assert arrays["patch_features"].shape == (len(source_indices),7,59,48)
    assert arrays["icsa_weights"].min() >= 0.5 and arrays["icsa_weights"].max() <= 1.5
    raw = np.concatenate(inputs)
    with torch.inference_mode():
        frame_log_power = model.spectral_conditioner.log_power_spectrum(
            torch.from_numpy(raw).to(device)).cpu().numpy()
    definition = (
        "branch_rms=sqrt(mean((a^2+b^2)/2)); temporal_rms=sqrt(mean((Ta^2+Tb^2)/2)); "
        "frequency_rms=sqrt(mean((frequency_scale*context)^2)); each relative RMS divides by max(branch_rms,1e-8). "
        "Reductions are over the 192 hidden channels. These are activation magnitudes, not attribution scores.")
    output_dir.mkdir(parents=True,exist_ok=True)
    output_path = output_dir/"saftformer_layer_visualizations.mat"
    payload = dict(arrays,iq=raw,frame_log_power=frame_log_power,
        labels=y[source_indices],snr=snr[source_indices],channel=channel[source_indices],
        predictions=predictions,logits=logits,classNames=np.asarray(class_names,dtype=object),
        channelNames=np.asarray(channel_names,dtype=object),sourceIndices=source_indices,
        sourceIndices1Based=source_indices+1,testPredictionIndices=positions,
        frameStarts=np.arange(59)*8,frameCentres=np.arange(59)*8+23.5,
        featureStageNames=np.asarray(["Embedding"]+[f"Block {i}" for i in range(1,7)],dtype=object),
        featureDefinition=np.asarray(["Patch-token embedding and post-residual outputs of Blocks 1-6, "
            "before the model's final LayerNorm. CLS/register tokens are excluded."],dtype=object),
        temporalKernelSizes=np.asarray(model.temporal_kernel_sizes),
        checkpointHash=np.asarray([checkpoint_hash],dtype=object),
        checkpointEpoch=int(checkpoint.get("epoch",provenance["checkpoint_epoch"])),
        selectionSeed=args.seed,normalization=np.asarray([config.data.normalize],dtype=object),
        responseDefinition=np.asarray([definition],dtype=object),
        sourceCheckpoint=np.asarray([str(checkpoint_path)],dtype=object),
        sourceDataset=np.asarray([str(dataset_path)],dtype=object))
    savemat(output_path,payload,do_compression=True)
    weights=arrays["icsa_weights"]
    report = dict(checkpoint_file=str(checkpoint_path),checkpoint_sha256=checkpoint_hash,
        checkpoint_epoch=int(payload["checkpointEpoch"]),config_file=str(config_path),dataset_file=str(dataset_path),
        provenance_file=str(provenance_path),output_file=str(output_path),device=str(device),precision="float32",
        selection_seed=args.seed,selection="Fixed class/channel/SNR cells. No prediction/activation filtering.",
        samples=len(source_indices),samples_per_cell=args.samples_per_cell,cells=cells,
        reference_prediction_mismatches=mismatches,repeated_forward_max_abs_logit_difference=native_max_difference,
        hook_free_forward_max_abs_logit_difference=hook_free_difference,
        icsa_weight_min=float(weights.min()),icsa_weight_max=float(weights.max()),
        icsa_weight_std=float(weights.std()),
        icsa_within_frame_std_mean=float(weights.std(axis=2).mean()),
        frequency_scales=arrays["frequency_scales"].tolist(),response_definition=definition,
        architecture_note=(
            "Full local conditions update patch tokens only. The final patch FFN has no path to the CLS "
            "classifier after attention. Its zero-initialized local parameters receive no classification gradient."),
        shapes={name:list(value.shape) for name,value in arrays.items()},
        include_hidden=args.include_hidden)
    for name in ("temporal_relative_rms","frequency_relative_rms"):
        values=arrays[name]
        report[name]=dict(min=float(values.min()),median=float(np.median(values)),max=float(values.max()),
                          layer_medians=np.median(values,axis=(0,2)).tolist())
    report_path=output_dir/"saftformer_layer_visualizations.json"
    report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Saved {output_path}; reference prediction mismatches={mismatches}.",flush=True)
    print(f"ICSA weight range [{weights.min():.5f},{weights.max():.5f}], std={weights.std():.5f}",flush=True)


if __name__ == "__main__":
    main()
