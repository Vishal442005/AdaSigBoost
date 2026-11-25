"""Out-of-Distribution Colored Noise & Sensor Drift Stress Tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit

from adasigboost import AdaSigBoostClassifier, AdaSigBoostRegressor
from benchmarks.baselines import (
    DeepTimeSeriesBaseline,
    RandomConvKernelBaseline,
    RawGBDTBaseline,
)
from benchmarks.data_loaders import load_benchmark_dataset
from benchmarks.synthetic import generate_colored_noise, generate_synthetic_benchmark


def _inject_drift(X: np.ndarray, ramp_scale: float = 2.0, step_scale: float = 1.5, seed: int = 0) -> np.ndarray:
    """Inject unseen slow linear sensor ramp + abrupt step offset per sample."""
    rng = np.random.default_rng(seed)
    X_3d = X[:, np.newaxis, :].copy() if X.ndim == 2 else X.copy()
    N, C, T = X_3d.shape
    t_grid = np.linspace(-0.5, 0.5, T)[np.newaxis, np.newaxis, :]
    std_per = np.std(X_3d, axis=-1, keepdims=True) + 1e-6
    signs = rng.choice([-1.0, 1.0], size=(N, C, 1))
    ramp = signs * ramp_scale * std_per * t_grid
    step = np.zeros_like(X_3d)
    step_pts = rng.integers(T // 4, (3 * T) // 4, size=(N, C))
    for i in range(N):
        for c in range(C):
            step[i, c, step_pts[i, c] :] = signs[i, c, 0] * step_scale * std_per[i, c, 0]
    out = X_3d + ramp + step
    return out[:, 0, :] if X.ndim == 2 else out


def _add_colored_ood_noise(X: np.ndarray, snr_db: float, beta: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X_3d = X[:, np.newaxis, :].copy() if X.ndim == 2 else X.copy()
    N, C, T = X_3d.shape
    for c in range(C):
        sigs = X_3d[:, c, :]
        sig_pow = np.mean(sigs**2, axis=-1, keepdims=True) + 1e-8
        noise = generate_colored_noise(rng, n_samples=N, signal_length=T, beta=beta)
        target_pow = sig_pow / (10.0 ** (snr_db / 10.0))
        X_3d[:, c, :] = sigs + np.sqrt(target_pow) * noise
    return X_3d[:, 0, :] if X.ndim == 2 else X_3d


def run_ood_stress_suite(
    output_dir: Path = Path("results"),
    fig_dir: Path = Path("paper/figures"),
    seed: int = 42,
) -> dict[str, Any]:
    """Train clean models and stress-test under unseen OOD noise & sensor drift."""
    ds = generate_synthetic_benchmark(n_samples=900, signal_length=512, snr_db=20.0, seed=seed)
    X_tr, y_tr = ds.X[:600], ds.y[:600]
    X_te_clean, y_te = ds.X[600:], ds.y[600:]

    models = {
        "AdaSigBoost": AdaSigBoostRegressor(
            n_estimators=22,
            n_modes=6,
            learning_rate=0.16,
            max_depth=4,
            lambda1=1.5,
            n_filter_steps=6,
            use_bic_stopping=False,
            random_state=seed,
        ),
        "MultiRocket+Ridge": RandomConvKernelBaseline(
            n_kernels=250, head="ridge", task="regression", seed=seed, alpha=10.0
        ),
        "LightGBM": RawGBDTBaseline(
            backend="lightgbm", task="regression", n_estimators=120, max_depth=4, random_state=seed
        ),
        "XGBoost": RawGBDTBaseline(
            backend="xgboost", task="regression", n_estimators=120, max_depth=4, random_state=seed
        ),
        "PatchTST": DeepTimeSeriesBaseline(
            arch="patchtst", epochs=20, lr=3e-3, task="regression", seed=seed
        ),
        "ModernTCN": DeepTimeSeriesBaseline(
            arch="moderntcn", epochs=20, lr=3e-3, task="regression", seed=seed
        ),
    }

    reg_tests = {
        "Clean (+20dB)": X_te_clean,
        "OOD Pink (+5dB, 1/f)": _add_colored_ood_noise(X_te_clean, snr_db=5.0, beta=1.0, seed=seed + 1),
        "OOD Drift Noise (0dB, 1/f^2)": _add_colored_ood_noise(
            X_te_clean, snr_db=0.0, beta=2.0, seed=seed + 2
        ),
        "OOD Ramp+Step Drift": _inject_drift(X_te_clean, ramp_scale=2.0, step_scale=1.5, seed=seed + 3),
    }

    synth_ood: dict[str, dict[str, float]] = {m: {} for m in models}
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        for cond, X_cond in reg_tests.items():
            rmse = float(root_mean_squared_error(y_te, mdl.predict(X_cond)))
            synth_ood[name][cond] = rmse
            print(f"[OOD Synth | {cond:28s}] {name:18s} -> RMSE={rmse:.4f}", flush=True)

    # Real-world FordA clean -> OOD colored noise & drift stress test
    ford = load_benchmark_dataset("FordA", seed=seed)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr_i, te_i = next(splitter.split(ford.X, ford.y))
    tr_i = tr_i[:600]
    te_i = te_i[:300]
    cls_models = {
        "AdaSigBoost": AdaSigBoostClassifier(
            n_estimators=20,
            n_modes=6,
            learning_rate=0.16,
            max_depth=4,
            lambda1=0.8,
            n_filter_steps=6,
            use_bic_stopping=False,
            random_state=seed,
        ),
        "MultiRocket+Ridge": RandomConvKernelBaseline(
            n_kernels=250, head="ridge", task="classification", seed=seed, alpha=10.0
        ),
        "LightGBM": RawGBDTBaseline(
            backend="lightgbm", task="classification", n_estimators=120, max_depth=4, random_state=seed
        ),
        "PatchTST": DeepTimeSeriesBaseline(
            arch="patchtst", epochs=20, lr=3e-3, task="classification", seed=seed
        ),
    }
    ford_tests = {
        "Clean": ford.X[te_i],
        "OOD Colored (+5dB)": _add_colored_ood_noise(ford.X[te_i], snr_db=5.0, beta=1.5, seed=seed + 10),
        "OOD Ramp+Step Drift": _inject_drift(ford.X[te_i], ramp_scale=1.5, step_scale=1.0, seed=seed + 11),
    }
    ford_ood: dict[str, dict[str, float]] = {m: {} for m in cls_models}
    for name, mdl in cls_models.items():
        mdl.fit(ford.X[tr_i], ford.y[tr_i])
        for cond, X_cond in ford_tests.items():
            probs = mdl.predict_proba(X_cond)[:, 1]
            auc = float(roc_auc_score(ford.y[te_i], probs))
            ford_ood[name][cond] = auc
            print(f"[OOD FordA | {cond:22s}] {name:18s} -> AUROC={auc:.4f}", flush=True)

    # Plot OOD degradation figure
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    conds_s = list(reg_tests.keys())
    x_s = np.arange(len(conds_s))
    for name, vals in synth_ood.items():
        lw = 2.4 if name == "AdaSigBoost" else 1.4
        axes[0].plot(x_s, [vals[c] for c in conds_s], marker="o", lw=lw, label=name)
    axes[0].set_xticks(x_s)
    axes[0].set_xticklabels(["Clean", "Pink\n+5dB", "1/f²\n0dB", "Ramp+\nStep"], fontsize=8)
    axes[0].set_ylabel("Test RMSE (Lower is Better)")
    axes[0].set_title("Synthetic OOD Noise & Sensor Drift")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    conds_f = list(ford_tests.keys())
    x_f = np.arange(len(conds_f))
    for name, vals in ford_ood.items():
        lw = 2.4 if name == "AdaSigBoost" else 1.4
        axes[1].plot(x_f, [vals[c] for c in conds_f], marker="s", lw=lw, label=name)
    axes[1].set_xticks(x_f)
    axes[1].set_xticklabels(["Clean", "Colored +5dB", "Ramp+Step Drift"], fontsize=8)
    axes[1].set_ylabel("Test AUROC (Higher is Better)")
    axes[1].set_title("FordA Acoustics OOD Stress Test")
    axes[1].grid(alpha=0.25, linestyle="--")
    axes[1].legend(frameon=False, fontsize=7.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig_ood_stress.pdf", bbox_inches="tight")

    out = {"synthetic_rmse": synth_ood, "forda_auroc": ford_ood}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ood_stress.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run_ood_stress_suite()
