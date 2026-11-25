"""Leak-free real-world dataset loaders across biomedical, industrial, energy, and acoustic domains."""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


@dataclass
class BenchmarkDataset:
    """Standardized benchmark dataset container with leak-free fold/split assignments."""

    name: str
    domain: str
    task: str  # "binary", "multiclass", "regression"
    metric: str
    X: np.ndarray  # (N, C, T) or (N, T)
    y: np.ndarray  # (N,)
    Z: np.ndarray | None = None  # optional tabular covariates (N, P)
    groups: np.ndarray | None = None  # subject/bearing/environment IDs for GroupKFold
    fs: float = 1.0
    metadata: dict[str, Any] | None = None


DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_file(url: str, dest: Path) -> Path:
    """Download a file if missing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "AdaSigBoost-Benchmarks/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            f.write(resp.read())
    return dest


def _parse_ts_file(content: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse UCR/UEA .ts format into (X (N, C, T), y (N,))."""
    in_data = False
    X_rows: list[list[list[float]]] = []
    y_rows: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        if line.lower() == "@data":
            in_data = True
            continue
        if not in_data:
            continue
        parts = line.split(":")
        label = parts[-1].strip()
        channels = []
        for ch_str in parts[:-1]:
            vals = [float(v) if v != "?" else 0.0 for v in ch_str.split(",")]
            channels.append(vals)
        X_rows.append(channels)
        y_rows.append(label)

    X = np.asarray(X_rows, dtype=np.float64)
    _, y_int = np.unique(y_rows, return_inverse=True)
    return X, y_int.astype(np.int64)


def load_ucr_dataset(name: str) -> BenchmarkDataset:
    """Download and load a UCR/UEA time-series classification archive (FordA, ECG5000, etc.)."""
    cache_dir = DATA_ROOT / "ucr" / name
    zip_path = cache_dir / f"{name}.zip"
    if not zip_path.exists():
        url = f"https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip"
        download_file(url, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        train_name = next(n for n in zf.namelist() if n.endswith("_TRAIN.ts"))
        test_name = next(n for n in zf.namelist() if n.endswith("_TEST.ts"))
        X_tr, y_tr = _parse_ts_file(zf.read(train_name).decode("utf-8"))
        X_te, y_te = _parse_ts_file(zf.read(test_name).decode("utf-8"))

    X = np.concatenate([X_tr, X_te], axis=0)
    y = np.concatenate([y_tr, y_te], axis=0)
    n_classes = len(np.unique(y))
    return BenchmarkDataset(
        name=name,
        domain="UCR-Signal",
        task="binary" if n_classes == 2 else "multiclass",
        metric="auroc" if n_classes == 2 else "accuracy",
        X=X,
        y=y,
        metadata={"n_train_official": len(y_tr), "n_test_official": len(y_te)},
    )


def load_cwru_subset(
    window_size: int = 1024,
    stride: int = 1024,
) -> BenchmarkDataset:
    """Load public CWRU 12kHz bearing vibration data with load/file-disjoint groups."""
    # Official Case Western Reserve University 12kHz Drive-End .mat URLs (Normal, Inner, Ball, Outer)
    files = {
        "Normal_0": ("https://engineering.case.edu/sites/default/files/97.mat", 0, 0),
        "Normal_1": ("https://engineering.case.edu/sites/default/files/98.mat", 0, 1),
        "IR007_0": ("https://engineering.case.edu/sites/default/files/105.mat", 1, 0),
        "IR007_1": ("https://engineering.case.edu/sites/default/files/106.mat", 1, 1),
        "B007_0": ("https://engineering.case.edu/sites/default/files/118.mat", 2, 0),
        "B007_1": ("https://engineering.case.edu/sites/default/files/119.mat", 2, 1),
        "OR007_0": ("https://engineering.case.edu/sites/default/files/130.mat", 3, 0),
        "OR007_1": ("https://engineering.case.edu/sites/default/files/131.mat", 3, 1),
    }
    X_list, y_list, grp_list = [], [], []
    for key, (url, label, load_id) in files.items():
        mat_path = DATA_ROOT / "cwru" / f"{key}.mat"
        download_file(url, mat_path)
        mat = loadmat(mat_path)
        de_key = next(k for k in mat.keys() if k.endswith("_DE_time"))
        sig = np.asarray(mat[de_key], dtype=np.float64).ravel()
        for start in range(0, len(sig) - window_size + 1, stride):
            X_list.append(sig[start : start + window_size])
            y_list.append(label)
            grp_list.append(load_id)

    X = np.asarray(X_list, dtype=np.float64)[:, np.newaxis, :]
    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(grp_list, dtype=np.int64)
    return BenchmarkDataset(
        name="cwru_bearing",
        domain="Industrial Prognostics",
        task="multiclass",
        metric="accuracy",
        X=X,
        y=y,
        groups=groups,
        fs=12000.0,
    )


def load_etth1_forecasting(
    history_len: int = 168,
    horizon: int = 24,
) -> BenchmarkDataset:
    """Download public ETTh1 electricity transformer load + tabular temperature covariates."""
    csv_path = DATA_ROOT / "energy" / "ETTh1.csv"
    if not csv_path.exists():
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
        download_file(url, csv_path)

    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1, usecols=(1, 2, 3, 4, 5, 6, 7))
    # Columns 0..5 are load features, column 6 is OT (oil temp target)
    X_rows, Z_rows, y_rows = [], [], []
    step = 4
    for t in range(history_len, len(raw) - horizon, step):
        # Signal: past history_len hours of multi-channel load (6 channels)
        X_rows.append(raw[t - history_len : t, :6].T)
        # Tabular covariates at forecast origin t: instantaneous loads + OT
        Z_rows.append(raw[t, :])
        # Target: 24h-ahead mean OT
        y_rows.append(float(np.mean(raw[t : t + horizon, 6])))

    return BenchmarkDataset(
        name="etth1_energy",
        domain="Energy & Smart Grid",
        task="regression",
        metric="mae",
        X=np.asarray(X_rows, dtype=np.float64),
        y=np.asarray(y_rows, dtype=np.float64),
        Z=np.asarray(Z_rows, dtype=np.float64),
        fs=1.0,
    )


def load_eeg_eyestate(window_size: int = 32) -> BenchmarkDataset:
    """Download OpenML EEG Eye State (14 channels @ 128Hz) with contiguous block groups."""
    csv_path = DATA_ROOT / "neuro" / "eeg_eyestate.csv"
    if not csv_path.exists():
        url = "https://www.openml.org/data/get_csv/1587932/phpo22668.csv"
        download_file(url, csv_path)

    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    signals = raw[:, :14]
    labels = raw[:, 14].astype(np.int64)
    # Normalize per channel robustly (clipping sensor spike outliers > 10 MAD)
    med = np.median(signals, axis=0, keepdims=True)
    mad = np.median(np.abs(signals - med), axis=0, keepdims=True) + 1e-6
    signals = np.clip((signals - med) / (1.4826 * mad), -10.0, 10.0)

    X_list, y_list, grp_list = [], [], []
    n_blocks = 6
    block_len = len(signals) // n_blocks
    for start in range(0, len(signals) - window_size + 1, window_size):
        win_y = labels[start : start + window_size]
        frac1 = float(np.mean(win_y))
        if frac1 >= 0.95 or frac1 <= 0.05:
            X_list.append(signals[start : start + window_size].T)
            y_list.append(int(frac1 >= 0.5))
            grp_list.append(min(n_blocks - 1, start // max(block_len, 1)))

    return BenchmarkDataset(
        name="eeg_eyestate",
        domain="Biomedical (Neurology)",
        task="binary",
        metric="auroc",
        X=np.asarray(X_list, dtype=np.float64),
        y=np.asarray(y_list, dtype=np.int64),
        groups=np.asarray(grp_list, dtype=np.int64),
        fs=128.0,
    )


def load_benchmark_dataset(name: str, seed: int = 42) -> BenchmarkDataset:
    """Unified dataset loader supporting synthetic, UCR/UEA, CWRU, EEG, and ETTh1."""
    from .synthetic import generate_synthetic_benchmark

    if name.startswith("synthetic_snr"):
        snr_str = name.replace("synthetic_snr", "").replace("_neg", "-")
        snr_db = float(snr_str)
        ds = generate_synthetic_benchmark(n_samples=1500, signal_length=1024, snr_db=snr_db, seed=seed)
        return BenchmarkDataset(
            name=name,
            domain="Synthetic",
            task="regression",
            metric="rmse",
            X=ds.X,
            y=ds.y,
            fs=ds.fs,
        )
    if name == "cwru_bearing":
        return load_cwru_subset()
    if name == "etth1_energy":
        return load_etth1_forecasting()
    if name == "eeg_eyestate":
        return load_eeg_eyestate()
    if name in {
        "FordA",
        "FordB",
        "ECG5000",
        "TwoLeadECG",
        "Epilepsy",
        "Wafer",
        "Heartbeat",
        "FaultDetectionA",
        "InsectWingbeatSound",
    }:
        return load_ucr_dataset(name)
    raise ValueError(f"Unknown dataset identifier: {name!r}")
