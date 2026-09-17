"""Load MATLAB-generated modulation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np


def is_hdf5_mat_file(path: Union[str, Path]) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    with path.open("rb") as file:
        header = file.read(32)
    if header.startswith(b"MATLAB 7.3 MAT-file"):
        return True

    try:
        import h5py
    except ImportError:
        return False

    try:
        with h5py.File(path, "r"):
            return True
    except OSError:
        return False


def _normalize_shape(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expected [N, 2, L] or [N, L, 2], got {x.shape}.")
    if x.shape[1] == 2:
        out = x
    elif x.shape[2] == 2:
        out = np.transpose(x, (0, 2, 1))
    else:
        raise ValueError(f"Could not infer I/Q axis from shape {x.shape}.")
    return out.astype(np.float32, copy=False)


def load_matlab_dataset(
    path: Union[str, Path],
    x_key: str = "X",
    y_key: str = "y",
    snr_key: Optional[str] = "snr",
    channel_key: Optional[str] = "channel",
    split_key: Optional[str] = "split",
    class_names: Optional[list[str]] = None,
    return_metadata: bool = False,
):
    """Load a .mat file exported from MATLAB.

    Expected variables:
        X: [N, 2, L] or [N, L, 2] I/Q samples
        y: integer labels or one-hot labels
        snr: optional SNR vector
        channel: optional zero-based channel labels
        split: optional 0=train, 1=val, 2=test vector
    """

    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("Install scipy to load MATLAB .mat datasets.") from exc

    data = loadmat(Path(path))
    if x_key not in data or y_key not in data:
        raise KeyError(f"MAT file must contain '{x_key}' and '{y_key}'.")

    x = _normalize_shape(data[x_key])
    y_raw = np.asarray(data[y_key])
    if y_raw.ndim > 1 and y_raw.shape[-1] > 1:
        y = np.argmax(y_raw, axis=-1)
    else:
        y = y_raw.reshape(-1)
    y = y.astype(np.int64)

    if snr_key and snr_key in data:
        snr = np.asarray(data[snr_key]).reshape(-1).astype(np.int64)
    else:
        snr = np.zeros(x.shape[0], dtype=np.int64)

    if class_names is None and "classNames" in data:
        class_names = _parse_matlab_class_names(data["classNames"])

    if class_names is None:
        n_classes = int(y.max()) + 1
        class_names = [f"class_{idx}" for idx in range(n_classes)]

    metadata = {}
    if channel_key and channel_key in data:
        metadata["channel"] = np.asarray(data[channel_key]).reshape(-1).astype(np.int64)
    else:
        metadata["channel"] = np.zeros(x.shape[0], dtype=np.int64)

    if split_key and split_key in data:
        metadata["split"] = _parse_split(data[split_key])
    else:
        metadata["split"] = None

    if "channelNames" in data:
        metadata["channel_names"] = _parse_matlab_class_names(data["channelNames"])
    else:
        n_channels = int(metadata["channel"].max()) + 1
        metadata["channel_names"] = [f"channel_{idx}" for idx in range(n_channels)]

    if "splitNames" in data:
        metadata["split_names"] = _parse_matlab_class_names(data["splitNames"])
    else:
        metadata["split_names"] = ["train", "val", "test"]

    if return_metadata:
        return x, y, snr, class_names, metadata
    return x, y, snr, class_names


def load_matlab_hdf5_metadata(
    path: Union[str, Path],
    y_key: str = "y",
    snr_key: Optional[str] = "snr",
    channel_key: Optional[str] = "channel",
    split_key: Optional[str] = "split",
    class_names: Optional[list[str]] = None,
    channel_names: Optional[list[str]] = None,
):
    """Read small metadata arrays from a MATLAB v7.3 file."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install h5py to load MATLAB -v7.3 datasets.") from exc

    with h5py.File(Path(path), "r") as h5:
        if "generationComplete" in h5:
            complete = bool(np.asarray(h5["generationComplete"]).reshape(-1)[0])
            if not complete:
                raise RuntimeError(
                    f"MATLAB dataset generation is incomplete: {path}"
                )

        y = _read_h5_vector(h5, y_key).astype(np.int64)
        if snr_key and snr_key in h5:
            snr = _read_h5_vector(h5, snr_key).astype(np.int64)
        else:
            snr = np.zeros(len(y), dtype=np.int64)

        if channel_key and channel_key in h5:
            channel = _read_h5_vector(h5, channel_key).astype(np.int64)
        else:
            channel = np.zeros(len(y), dtype=np.int64)

        if split_key and split_key in h5:
            split = _read_h5_vector(h5, split_key).astype(np.int64)
        else:
            split = None

        stored_class_names = (
            _read_h5_string_cell(h5, "classNames") if "classNames" in h5 else None
        )
        stored_channel_names = (
            _read_h5_string_cell(h5, "channelNames") if "channelNames" in h5 else None
        )

        if class_names is None:
            class_names = stored_class_names
        elif stored_class_names is not None and list(class_names) != stored_class_names:
            raise ValueError(
                "Configured modulation classes do not match classNames stored in "
                f"{path}. Regenerate the dataset or use its matching config."
            )

        if channel_names is None:
            channel_names = stored_channel_names
        elif stored_channel_names is not None and list(channel_names) != stored_channel_names:
            raise ValueError(
                "Configured channels do not match channelNames stored in "
                f"{path}. Regenerate the dataset or use its matching config."
            )

    if class_names is None:
        class_names = [f"class_{idx}" for idx in range(int(y.max()) + 1)]
    if channel_names is None:
        channel_names = [f"channel_{idx}" for idx in range(int(channel.max()) + 1)]

    metadata = {
        "channel": channel,
        "split": split,
        "channel_names": channel_names,
        "split_names": ["train", "val", "test"],
    }
    return y, snr, class_names, metadata


def _read_h5_vector(h5, key: str) -> np.ndarray:
    arr = np.asarray(h5[key])
    return arr.reshape(-1, order="F")


def _read_h5_string_cell(h5, key: str) -> list[str]:
    raw = h5[key]
    values = np.asarray(raw)
    names = []
    if values.dtype == object:
        for ref in values.reshape(-1, order="F"):
            names.append(_decode_h5_string(np.asarray(h5[ref])))
    else:
        if values.ndim == 2:
            for idx in range(values.shape[1]):
                names.append(_decode_h5_string(values[:, idx]))
        else:
            names.append(_decode_h5_string(values))
    return [name for name in names if name]


def _decode_h5_string(values: np.ndarray) -> str:
    arr = np.asarray(values).reshape(-1, order="F")
    chars = []
    for value in arr:
        code = int(value)
        if code:
            chars.append(chr(code))
    return "".join(chars)


def _parse_matlab_class_names(raw) -> list[str]:
    names = []
    arr = np.asarray(raw).squeeze()
    if arr.ndim == 0:
        arr = np.asarray([arr.item()])
    for item in arr:
        value = np.asarray(item).squeeze()
        if value.dtype.kind in {"U", "S"}:
            names.append("".join(value.tolist()) if value.ndim else str(value.item()))
        elif value.dtype == object:
            flat = value.reshape(-1)
            text = "".join(str(x) for x in flat)
            names.append(text)
        else:
            names.append(str(value))
    return names


def _parse_split(raw) -> np.ndarray:
    arr = np.asarray(raw).squeeze()
    if arr.dtype.kind in {"i", "u", "f"}:
        return arr.reshape(-1).astype(np.int64)

    names = _parse_matlab_class_names(raw)
    mapping = {"train": 0, "val": 1, "valid": 1, "validation": 1, "test": 2}
    return np.asarray([mapping[str(name).strip().lower()] for name in names], dtype=np.int64)
