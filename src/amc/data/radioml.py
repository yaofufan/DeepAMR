"""Load DeepSig RadioML datasets into numpy arrays."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .constants import RML2016_10A_CLASSES, RML2016_10B_CLASSES, RML2018_01A_CLASSES


def _normalize_shape(x: np.ndarray) -> np.ndarray:
    """Return samples as float32 with shape [N, 2, L]."""

    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expected a 3D signal array, got shape {x.shape}.")
    if x.shape[1] == 2:
        out = x
    elif x.shape[2] == 2:
        out = np.transpose(x, (0, 2, 1))
    else:
        raise ValueError(f"Could not infer I/Q axis from shape {x.shape}.")
    return out.astype(np.float32, copy=False)


def _load_pickle_dict(path: Path, class_names: Optional[list[str]] = None):
    with path.open("rb") as f:
        try:
            data = pickle.load(f, encoding="latin1")
        except TypeError:
            data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected a RadioML 2016 pickle dictionary.")

    mods = sorted({key[0] for key in data.keys()})
    snrs = sorted({int(key[1]) for key in data.keys()})
    if class_names is None:
        if "AM-SSB" in mods:
            class_names = RML2016_10A_CLASSES
        else:
            class_names = RML2016_10B_CLASSES
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    xs, ys, snr_values = [], [], []
    for mod in class_names:
        for snr in snrs:
            key = (mod, snr)
            if key not in data:
                continue
            arr = _normalize_shape(data[key])
            xs.append(arr)
            ys.append(np.full(arr.shape[0], class_to_idx[mod], dtype=np.int64))
            snr_values.append(np.full(arr.shape[0], snr, dtype=np.int64))

    if not xs:
        raise ValueError(f"No samples found in {path}.")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(snr_values), class_names


def _load_hdf5(path: Path, class_names: Optional[list[str]] = None):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install h5py to load RadioML 2018 HDF5 files.") from exc

    with h5py.File(path, "r") as h5:
        x_key = "X" if "X" in h5 else next(iter(h5.keys()))
        x = _normalize_shape(np.asarray(h5[x_key]))
        if "Y" in h5:
            y_raw = np.asarray(h5["Y"])
            y = np.argmax(y_raw, axis=1).astype(np.int64)
        elif "y" in h5:
            y = np.asarray(h5["y"]).reshape(-1).astype(np.int64)
        else:
            raise ValueError("HDF5 file needs a Y or y label dataset.")

        if "Z" in h5:
            snr = np.asarray(h5["Z"]).reshape(-1).astype(np.int64)
        elif "snr" in h5:
            snr = np.asarray(h5["snr"]).reshape(-1).astype(np.int64)
        else:
            snr = np.zeros(x.shape[0], dtype=np.int64)

    return x, y, snr, class_names or RML2018_01A_CLASSES


def load_radioml(path: Union[str, Path], class_names: Optional[list[str]] = None):
    """Load RadioML 2016 pickle/dat or RadioML 2018 HDF5 data.

    Returns:
        Tuple of (x, y, snr, class_names), where x has shape [N, 2, L].
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".dat"}:
        return _load_pickle_dict(path, class_names)
    if suffix in {".h5", ".hdf5"}:
        return _load_hdf5(path, class_names)
    raise ValueError(f"Unsupported RadioML file extension: {path.suffix}")
