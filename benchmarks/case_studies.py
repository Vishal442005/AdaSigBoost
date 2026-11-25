"""Phase 6 Interpretability Case Studies (CWRU Bearing, Mode Deletion AUC, Low-Data Curve)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, root_mean_squared_error

from adasigboost import AdaSigBoostClassifier, AdaSigBoostRegressor
from adasigboost.interpretability import (
    compute_mode_deletion_auc,
    extract_learned_frequencies,
    plot_mode_decomposition,
    spectral_frequency_alignment_error,
    summarize_stage_modes,
)
from benchmarks.baselines import (
    LearnableFilterNeuralBaseline,
    RandomConvKernelBaseline,
    RawGBDTBaseline,
)
from benchmarks.data_loaders import load_benchmark_dataset
from benchmarks.synthetic import generate_synthetic_benchmark


def run_cwru_case_study(
    output_dir: Path = Path("results"),
    fig_dir: Path = Path("paper/figures"),
) -> dict[str, Any]:
    """Train AdaSigBoost on load-disjoint CWRU 12kHz bearing signals & inspect learned modes."""
    ds = load_benchmark_dataset("cwru_bearing")
    assert ds.groups is not None
    tr_mask = ds.groups == 0
    te_mask = ds.groups == 1

    clf = AdaSigBoostClassifier(
        n_estimators=20,
        n_modes=6,
        learning_rate=0.15,
        max_depth=4,
        lambda1=1.0,
        n_filter_steps=8,
        use_bic_stopping=False,
        random_state=42,
    )
    clf.fit(ds.X[tr_mask], ds.y[tr_mask])
    preds = clf.predict(ds.X[te_mask])
    acc = float(accuracy_score(ds.y[te_mask], preds))

    # Mode deletion faithfulness curve
    del_res = compute_mode_deletion_auc(
        clf,
        ds.X[te_mask],
        ds.y[te_mask],
        metric_fn=lambda yt, F: accuracy_score(yt, np.argmax(F, axis=1)),
    )

    summary = summarize_stage_modes(clf, fs=ds.fs)
    top_modes = extract_learned_frequencies(clf, fs=ds.fs)
    top_modes.sort(key=lambda r: r["weight_norm"], reverse=True)
    # CWRU SKF 6205-2RS @ 1772-1797 RPM characteristic frequencies & bearing housing carrier harmonics
    cwru_targets_hz = [107.4, 162.2, 2800.0, 3500.0]
    hz_alignment = spectral_frequency_alignment_error(clf, cwru_targets_hz, fs=ds.fs)

    # Outer-race fault sample visualization (label == 3)
    or_idx = int(np.flatnonzero(ds.y[te_mask] == 3)[0])
    plot_mode_decomposition(
        clf,
        ds.X[te_mask][or_idx],
        fs=ds.fs,
        stage_idx=0,
        save_path=fig_dir / "fig_cwru_modes.pdf",
    )

    out = {
        "load_disjoint_accuracy": acc,
        "mode_deletion_auc": del_res,
        "hz_alignment_error": hz_alignment,
        "mode_summary": {str(k): v for k, v in summary.items()},
        "top_stage_modes": top_modes[:12],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cwru_case_study.json").write_text(json.dumps(out, indent=2))
    print(f"[CWRU Case Study] Load-disjoint Acc={acc:.4f}, Deletion AUC={del_res['deletion_auc']:.4f}", flush=True)
    return out


def run_low_data_curve(
    output_dir: Path = Path("results"),
    fig_dir: Path = Path("paper/figures"),
) -> dict[str, Any]:
    """Evaluate sample efficiency for N in {100, 200, 400, 600} at +15 dB SNR."""
    ds = generate_synthetic_benchmark(n_samples=900, signal_length=512, snr_db=15.0, seed=7)
    X_te, y_te = ds.X[600:], ds.y[600:]
    sizes = [100, 200, 400, 600]
    models = ["AdaSigBoost", "MultiRocket+Ridge", "LightGBM (Raw)", "LearnableFilter+MLP"]
    curve: dict[str, list[float]] = {m: [] for m in models}

    for n in sizes:
        X_tr, y_tr = ds.X[:n], ds.y[:n]
        cands = {
            "AdaSigBoost": AdaSigBoostRegressor(
                n_estimators=20,
                n_modes=6,
                learning_rate=0.18,
                max_depth=4,
                lambda1=2.0,
                n_filter_steps=6,
                use_bic_stopping=False,
                random_state=7,
            ),
            "MultiRocket+Ridge": RandomConvKernelBaseline(n_kernels=250, head="ridge", task="regression", seed=7, alpha=10.0),
            "LightGBM (Raw)": RawGBDTBaseline(backend="lightgbm", task="regression", n_estimators=100, max_depth=4, random_state=7),
            "LearnableFilter+MLP": LearnableFilterNeuralBaseline(n_filters=8, epochs=30, lr=1e-2, task="regression", seed=7),
        }
        for name, mdl in cands.items():
            mdl.fit(X_tr, y_tr)
            rmse = float(root_mean_squared_error(y_te, mdl.predict(X_te)))
            curve[name].append(rmse)
            print(f"[Low-Data N={n:3d}] {name:22s} -> RMSE={rmse:.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for name, vals in curve.items():
        lw = 2.4 if name == "AdaSigBoost" else 1.5
        ax.plot(sizes, vals, marker="o", lw=lw, label=name)
    ax.set_xlabel("Training Sample Size (N)")
    ax.set_ylabel("Test RMSE (Lower is Better)")
    ax.set_title("Low-Data Regime Learning Curve (+15 dB SNR)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig_low_data.pdf", bbox_inches="tight")

    out = {"sizes": sizes, "rmse_by_model": curve}
    (output_dir / "low_data_curve.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run_cwru_case_study()
    run_low_data_curve()
