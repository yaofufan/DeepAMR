"""PyTorch datasets and dataloader construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from amc.config import DataConfig, resolve_src_path
from .matlab import is_hdf5_mat_file, load_matlab_dataset, load_matlab_hdf5_metadata
from .radioml import load_radioml


class SignalDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        snr: Optional[np.ndarray] = None,
        channel: Optional[np.ndarray] = None,
        normalize: str = "rms",
    ) -> None:
        self.x = self._normalize(x, normalize)
        self.y = y.astype(np.int64)
        self.snr = np.zeros(len(y), dtype=np.int64) if snr is None else snr.astype(np.int64)
        self.channel = (
            np.zeros(len(y), dtype=np.int64)
            if channel is None
            else channel.astype(np.int64)
        )

    @staticmethod
    def _normalize(x: np.ndarray, mode: str) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        if mode == "none":
            return x
        if mode == "rms":
            rms = np.sqrt(np.mean(x * x, axis=(1, 2), keepdims=True) + 1e-8)
            return x / rms
        if mode == "zscore":
            mean = np.mean(x, axis=(1, 2), keepdims=True)
            std = np.std(x, axis=(1, 2), keepdims=True) + 1e-8
            return (x - mean) / std
        raise ValueError(f"Unknown normalization mode: {mode}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return {
            "x": torch.from_numpy(self.x[index]),
            "y": torch.tensor(self.y[index], dtype=torch.long),
            "snr": torch.tensor(self.snr[index], dtype=torch.long),
            "channel": torch.tensor(self.channel[index], dtype=torch.long),
        }


class H5MatSignalDataset(Dataset):
    """Lazy reader for large MATLAB -v7.3 datasets."""

    def __init__(
        self,
        path: str,
        x_key: str,
        y: np.ndarray,
        snr: np.ndarray,
        channel: np.ndarray,
        normalize: str = "rms",
    ) -> None:
        self.path = path
        self.x_key = x_key
        self.y = y.astype(np.int64)
        self.snr = snr.astype(np.int64)
        self.channel = channel.astype(np.int64)
        self.normalize = normalize
        self._h5 = None

    def _file(self):
        if self._h5 is None:
            import h5py

            self._h5 = h5py.File(
                self.path,
                "r",
                rdcc_nbytes=64 * 1024 * 1024,
                rdcc_nslots=100_003,
            )
        return self._h5

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = self._read_x(index)
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(self.y[index], dtype=torch.long),
            "snr": torch.tensor(self.snr[index], dtype=torch.long),
            "channel": torch.tensor(self.channel[index], dtype=torch.long),
        }

    def __getitems__(self, indices):
        """Read a complete DataLoader batch with one HDF5 selection."""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            return []

        order = np.argsort(indices)
        sorted_indices = indices[order]
        if np.any(sorted_indices[1:] == sorted_indices[:-1]):
            return [self[int(index)] for index in indices]

        ds = self._file()[self.x_key]
        n = len(self.y)
        sample_first = ds.shape[0] == n
        sample_last = ds.shape[-1] == n
        if not sample_first and not sample_last:
            raise ValueError(f"Cannot infer sample axis for HDF5 X shape {ds.shape}.")

        split_points = np.flatnonzero(np.diff(sorted_indices) != 1) + 1
        index_runs = np.split(sorted_indices, split_points)
        parts = []
        for run in index_runs:
            sample_slice = slice(int(run[0]), int(run[-1]) + 1)
            if sample_first:
                part = np.asarray(ds[sample_slice, :, :])
                if part.shape[1] == 2:
                    pass
                elif part.shape[2] == 2:
                    part = np.transpose(part, (0, 2, 1))
                else:
                    raise ValueError(
                        f"Cannot infer I/Q axis for HDF5 X shape {ds.shape}."
                    )
            else:
                part = np.asarray(ds[:, :, sample_slice])
                if part.shape[1] == 2:
                    part = np.transpose(part, (2, 1, 0))
                elif part.shape[0] == 2:
                    part = np.transpose(part, (2, 0, 1))
                else:
                    raise ValueError(
                        f"Cannot infer I/Q axis for HDF5 X shape {ds.shape}."
                    )
            parts.append(part)
        batch = np.concatenate(parts, axis=0)

        restore_order = np.argsort(order)
        batch = self._normalize_batch(batch[restore_order])
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        return [
            {
                "x": torch.from_numpy(batch[position]),
                "y": torch.tensor(self.y[index], dtype=torch.long),
                "snr": torch.tensor(self.snr[index], dtype=torch.long),
                "channel": torch.tensor(self.channel[index], dtype=torch.long),
            }
            for position, index in enumerate(indices)
        ]

    def _read_x(self, index: int) -> np.ndarray:
        ds = self._file()[self.x_key]
        shape = ds.shape
        n = len(self.y)
        if shape[0] == n:
            arr = np.asarray(ds[index, :, :])
        elif shape[-1] == n:
            arr = np.asarray(ds[:, :, index])
        else:
            raise ValueError(f"Cannot infer sample axis for HDF5 X shape {shape}.")
        arr = np.squeeze(arr).astype(np.float32, copy=False)
        if arr.ndim != 2:
            raise ValueError(f"Expected one sample to be 2D, got {arr.shape}.")
        if arr.shape[0] == 2:
            out = arr
        elif arr.shape[1] == 2:
            out = arr.T
        else:
            raise ValueError(f"Cannot infer I/Q axis for HDF5 sample shape {arr.shape}.")
        return self._normalize(out)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return x
        if self.normalize == "rms":
            return x / np.sqrt(np.mean(x * x) + 1e-8)
        if self.normalize == "zscore":
            return (x - np.mean(x)) / (np.std(x) + 1e-8)
        raise ValueError(f"Unknown normalization mode: {self.normalize}")

    def _normalize_batch(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        if self.normalize == "none":
            return x
        if self.normalize == "rms":
            rms = np.sqrt(np.mean(x * x, axis=(1, 2), keepdims=True) + 1e-8)
            return x / rms
        if self.normalize == "zscore":
            mean = np.mean(x, axis=(1, 2), keepdims=True)
            std = np.std(x, axis=(1, 2), keepdims=True) + 1e-8
            return (x - mean) / std
        raise ValueError(f"Unknown normalization mode: {self.normalize}")


class H5BlockBatchSampler(Sampler[list[int]]):
    """Shuffle short contiguous HDF5 blocks instead of individual disk offsets."""

    def __init__(
        self,
        indices: np.ndarray,
        batch_size: int,
        block_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if batch_size < 1 or block_size < 1:
            raise ValueError("batch_size and hdf5_block_size must be positive.")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        sorted_indices = np.sort(np.asarray(indices, dtype=np.int64))
        split_points = np.flatnonzero(np.diff(sorted_indices) != 1) + 1
        runs = np.split(sorted_indices, split_points)
        self.blocks = [
            run[start:start + block_size]
            for run in runs
            for start in range(0, len(run), block_size)
        ]
        self.num_samples = len(sorted_indices)

    def __iter__(self) -> Iterator[list[int]]:
        block_order = np.arange(len(self.blocks))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(block_order)
            self.epoch += 1
        ordered = np.concatenate([self.blocks[index] for index in block_order])
        for start in range(0, self.num_samples, self.batch_size):
            yield ordered[start:start + self.batch_size].tolist()

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]
    channel_names: list[str]


def _split_indices(
    y: np.ndarray,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    snr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    groups = []
    for cls in np.unique(y):
        if snr is None:
            groups.append((cls, None))
        else:
            groups.extend((cls, level) for level in np.unique(snr[y == cls]))

    for cls, level in groups:
        if level is None:
            indices = np.where(y == cls)[0]
        else:
            indices = np.where((y == cls) & (snr == level))[0]
        if len(indices) == 0:
            continue
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * train_fraction))
        n_val = int(round(n * val_fraction))
        train_idx.append(indices[:n_train])
        val_idx.append(indices[n_train:n_train + n_val])
        test_idx.append(indices[n_train + n_val:])
    return (
        rng.permutation(np.concatenate(train_idx)),
        rng.permutation(np.concatenate(val_idx)),
        rng.permutation(np.concatenate(test_idx)),
    )


def _indices_from_predefined_split(split: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = split.reshape(-1)
    return (
        np.where(split == 0)[0],
        np.where(split == 1)[0],
        np.where(split == 2)[0],
    )


def _load_arrays(config: DataConfig):
    class_names = config.class_names
    if config.path is None:
        raise ValueError("data.path is required for MATLAB and RadioML datasets.")
    if config.dataset.lower() in {"radioml", "radioml2016", "radioml2018"}:
        x, y, snr, class_names = load_radioml(config.path, class_names)
        metadata = {
            "channel": np.zeros(x.shape[0], dtype=np.int64),
            "split": None,
            "channel_names": ["mixed"],
        }
        return x, y, snr, class_names, metadata
    if config.dataset.lower() in {"matlab", "mat"}:
        return load_matlab_dataset(
            config.path,
            x_key=config.mat_x_key,
            y_key=config.mat_y_key,
            snr_key=config.mat_snr_key,
            channel_key=config.mat_channel_key,
            split_key=config.mat_split_key,
            class_names=class_names,
            return_metadata=True,
        )
    raise ValueError(f"Unknown dataset type: {config.dataset}")


def build_dataloaders(config: DataConfig) -> DataBundle:
    if config.path is None:
        raise ValueError("data.path is required for MATLAB and RadioML datasets.")
    data_path = resolve_src_path(config.path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path.resolve()}")
    config = replace(config, path=str(data_path))

    if (
        config.dataset.lower() in {"matlab", "mat"}
        and is_hdf5_mat_file(config.path)
    ):
        y, snr, class_names, metadata = load_matlab_hdf5_metadata(
            config.path,
            y_key=config.mat_y_key,
            snr_key=config.mat_snr_key,
            channel_key=config.mat_channel_key,
            split_key=config.mat_split_key,
            class_names=config.class_names,
            channel_names=config.channel_names,
        )
        dataset = H5MatSignalDataset(
            config.path,
            config.mat_x_key,
            y,
            snr,
            metadata["channel"],
            normalize=config.normalize,
        )
        if metadata.get("split") is not None:
            train_idx, val_idx, test_idx = _indices_from_predefined_split(metadata["split"])
        else:
            train_idx, val_idx, test_idx = _split_indices(
                y,
                config.train_fraction,
                config.val_fraction,
                config.seed,
                snr,
            )
        return _make_bundle(dataset, train_idx, val_idx, test_idx, class_names, metadata, config)

    x, y, snr, class_names, metadata = _load_arrays(config)
    dataset = SignalDataset(
        x,
        y,
        snr,
        channel=metadata.get("channel"),
        normalize=config.normalize,
    )
    if metadata.get("split") is not None:
        train_idx, val_idx, test_idx = _indices_from_predefined_split(metadata["split"])
    else:
        train_idx, val_idx, test_idx = _split_indices(
            y,
            config.train_fraction,
            config.val_fraction,
            config.seed,
            snr,
        )

    return _make_bundle(dataset, train_idx, val_idx, test_idx, class_names, metadata, config)


def _make_bundle(
    dataset: Dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    class_names: list[str],
    metadata: Dict[str, object],
    config: DataConfig,
) -> DataBundle:
    loader_kwargs: Dict[str, object] = {
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if config.num_workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": config.persistent_workers,
                "prefetch_factor": config.prefetch_factor,
            }
        )
    if isinstance(dataset, H5MatSignalDataset):
        train_loader = DataLoader(
            dataset,
            batch_sampler=H5BlockBatchSampler(
                train_idx,
                config.batch_size,
                config.hdf5_block_size,
                shuffle=True,
                seed=config.seed,
            ),
            **loader_kwargs,
        )
        val_loader = DataLoader(
            dataset,
            batch_sampler=H5BlockBatchSampler(
                val_idx,
                config.batch_size,
                config.hdf5_block_size,
                shuffle=False,
                seed=config.seed,
            ),
            **loader_kwargs,
        )
        test_loader = DataLoader(
            dataset,
            batch_sampler=H5BlockBatchSampler(
                test_idx,
                config.batch_size,
                config.hdf5_block_size,
                shuffle=False,
                seed=config.seed,
            ),
            **loader_kwargs,
        )
    else:
        regular_kwargs = {"batch_size": config.batch_size, **loader_kwargs}
        train_loader = DataLoader(
            Subset(dataset, train_idx), shuffle=True, **regular_kwargs
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), shuffle=False, **regular_kwargs
        )
        test_loader = DataLoader(
            Subset(dataset, test_idx), shuffle=False, **regular_kwargs
        )
    return DataBundle(
        train_loader,
        val_loader,
        test_loader,
        class_names,
        metadata.get("channel_names", ["unknown"]),
    )
