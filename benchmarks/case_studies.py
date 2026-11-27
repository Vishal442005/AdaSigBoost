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
    compute_mode_insertion_auc,
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

    # Mode deletion & insertion faithfulness curves
    del_res = compute_mode_deletion_auc(
        clf,
        ds.X[te_mask],
        ds.y[te_mask],
        metric_fn=lambda yt, F: accuracy_score(yt, np.argmax(F, axis=1)),
    )
    ins_res = compute_mode_insertion_auc(
        clf,
        ds.X[te_mask],
        ds.y[te_mask],
        metric_fn=lambda yt, F: accuracy_score(yt, np.argmax(F, axis=1)),
    )

    summary = summarize_stage_modes(clf, fs=ds.fs)
    top_modes = extract_learned_frequencies(clf, fs=ds.fs)
    top_modes.sort(key=lambda r: r["weight_norm"], reverse=True)
    atom_shares: dict[str, float] = {}
    for rec in top_modes:
        atom_shares[rec["atom_type"]] = atom_shares.get(rec["atom_type"], 0.0) + rec["weight_norm"]
    tot_w = sum(atom_shares.values()) or 1.0
    atom_shares = {k: v / tot_w for k, v in atom_shares.items()}
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
        "mode_insertion_auc": ins_res,
        "hz_alignment_error": hz_alignment,
        "atom_weight_share": atom_shares,
        "mode_summary": {str(k): v for k, v in summary.items()},
        "top_stage_modes": top_modes[:12],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cwru_case_study.json").write_text(json.dumps(out, indent=2))
    print(
        f"[CWRU Case Study] Load-disjoint Acc={acc:.4f}, Deletion AUC={del_res['deletion_auc']:.4f}, Insertion AUC={ins_res['insertion_auc']:.4f}",
        flush=True,
    )
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


def run_ecg_case_study(
    output_dir: Path = Path("results"),
    fig_dir: Path = Path("paper/figures"),
) -> dict[str, Any]:
    """Train AdaSigBoost on ECG5000 arrhythmia beats & compute clinical mode deletion AUC."""
    from sklearn.model_selection import StratifiedShuffleSplit

    ds = load_benchmark_dataset("ECG5000")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(ds.X, ds.y))
    tr_idx = tr_idx[:800]
    te_idx = te_idx[:300]

    clf = AdaSigBoostClassifier(
        n_estimators=24,
        n_modes=6,
        learning_rate=0.16,
        max_depth=4,
        lambda1=0.8,
        n_filter_steps=6,
        use_bic_stopping=False,
        random_state=42,
    )
    clf.fit(ds.X[tr_idx], ds.y[tr_idx])
    acc = float(accuracy_score(ds.y[te_idx], clf.predict(ds.X[te_idx])))

    del_res = compute_mode_deletion_auc(
        clf,
        ds.X[te_idx],
        ds.y[te_idx],
        metric_fn=lambda yt, F: accuracy_score(yt, np.argmax(F, axis=1)),
    )
    ins_res = compute_mode_insertion_auc(
        clf,
        ds.X[te_idx],
        ds.y[te_idx],
        metric_fn=lambda yt, F: accuracy_score(yt, np.argmax(F, axis=1)),
    )
    ecg_modes = extract_learned_frequencies(clf, fs=140.0)
    ecg_atoms: dict[str, float] = {}
    for rec in ecg_modes:
        ecg_atoms[rec["atom_type"]] = ecg_atoms.get(rec["atom_type"], 0.0) + rec["weight_norm"]
    tot_e = sum(ecg_atoms.values()) or 1.0
    ecg_atoms = {k: v / tot_e for k, v in ecg_atoms.items()}
    # Plot abnormal class sample (label != 0)
    abn_candidates = np.flatnonzero(ds.y[te_idx] != 0)
    abn_idx = int(abn_candidates[0]) if len(abn_candidates) > 0 else 0
    plot_mode_decomposition(
        clf,
        ds.X[te_idx][abn_idx],
        fs=140.0,
        stage_idx=0,
        save_path=fig_dir / "fig_ecg_modes.pdf",
    )

    out = {
        "test_accuracy": acc,
        "mode_deletion_auc": del_res,
        "mode_insertion_auc": ins_res,
        "atom_weight_share": ecg_atoms,
        "mode_summary": {str(k): v for k, v in summarize_stage_modes(clf, fs=140.0).items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ecg_case_study.json").write_text(json.dumps(out, indent=2))
    print(
        f"[ECG5000 Case Study] Acc={acc:.4f}, Deletion AUC={del_res['deletion_auc']:.4f}, Insertion AUC={ins_res['insertion_auc']:.4f}",
        flush=True,
    )
    return out


def run_multimodal_ablation(
    output_dir: Path = Path("results"),
) -> dict[str, Any]:
    """Compare Signal+Tabular vs Signal-Only vs Tabular-Only on etth1_energy."""
    from sklearn.metrics import mean_absolute_error

    ds = load_benchmark_dataset("etth1_energy")
    assert ds.Z is not None
    N = 900
    tr_idx = np.arange(0, 600)
    te_idx = np.arange(600, N)

    m_both = AdaSigBoostRegressor(
        n_estimators=35,
        n_modes=6,
        learning_rate=0.15,
        max_depth=4,
        lambda1=1.0,
        n_filter_steps=6,
        use_bic_stopping=False,
        random_state=42,
    )
    m_both.fit(ds.X[tr_idx], ds.y[tr_idx], Z=ds.Z[tr_idx])
    mae_both = float(mean_absolute_error(ds.y[te_idx], m_both.predict(ds.X[te_idx], Z=ds.Z[te_idx])))

    m_sig = AdaSigBoostRegressor(
        n_estimators=35,
        n_modes=6,
        learning_rate=0.15,
        max_depth=4,
        lambda1=1.0,
        n_filter_steps=6,
        use_bic_stopping=False,
        random_state=42,
    )
    m_sig.fit(ds.X[tr_idx], ds.y[tr_idx])
    mae_sig = float(mean_absolute_error(ds.y[te_idx], m_sig.predict(ds.X[te_idx])))

    import lightgbm as lgb

    m_tab = lgb.LGBMRegressor(n_estimators=100, max_depth=4, verbose=-1, random_state=42)
    m_tab.fit(ds.Z[tr_idx], ds.y[tr_idx])
    mae_tab = float(mean_absolute_error(ds.y[te_idx], m_tab.predict(ds.Z[te_idx])))

    # Also evaluate genuine Signal + Tabular complementary physics fusion (signal burst + RPM/temp calibration Z)
    rng = np.random.default_rng(42)
    syn = generate_synthetic_benchmark(n_samples=600, signal_length=256, snr_db=15.0, seed=42)
    Z_cov = rng.normal(size=(600, 4))
    y_joint = syn.y + 2.2 * np.maximum(Z_cov[:, 0], 0.0) - 1.4 * np.abs(Z_cov[:, 1])
    X_tr_j, X_te_j = syn.X[:400], syn.X[400:]
    Z_tr_j, Z_te_j = Z_cov[:400], Z_cov[400:]
    y_tr_j, y_te_j = y_joint[:400], y_joint[400:]

    j_both = AdaSigBoostRegressor(
        n_estimators=28,
        n_modes=6,
        learning_rate=0.16,
        max_depth=4,
        lambda1=1.2,
        n_filter_steps=6,
        use_bic_stopping=False,
        random_state=42,
    ).fit(X_tr_j, y_tr_j, Z=Z_tr_j)
    mae_j_both = float(mean_absolute_error(y_te_j, j_both.predict(X_te_j, Z=Z_te_j)))

    j_sig = AdaSigBoostRegressor(
        n_estimators=28,
        n_modes=6,
        learning_rate=0.16,
        max_depth=4,
        lambda1=1.2,
        n_filter_steps=6,
        use_bic_stopping=False,
        random_state=42,
    ).fit(X_tr_j, y_tr_j)
    mae_j_sig = float(mean_absolute_error(y_te_j, j_sig.predict(X_te_j)))

    j_tab = lgb.LGBMRegressor(n_estimators=100, max_depth=4, verbose=-1, random_state=42)
    j_tab.fit(Z_tr_j, y_tr_j)
    mae_j_tab = float(mean_absolute_error(y_te_j, j_tab.predict(Z_te_j)))

    out = {
        "etth1_energy": {
            "AdaSigBoost (Signal + Z)": mae_both,
            "AdaSigBoost (Signal Only)": mae_sig,
            "LightGBM (Z Only)": mae_tab,
        },
        "joint_signal_covariate": {
            "AdaSigBoost (Signal + Z)": mae_j_both,
            "AdaSigBoost (Signal Only)": mae_j_sig,
            "LightGBM (Z Only)": mae_j_tab,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "multimodal_ablation.json").write_text(json.dumps(out, indent=2))
    print(f"[Multimodal etth1] Signal+Z={mae_both:.4f} | Signal-Only={mae_sig:.4f} | Z-Only={mae_tab:.4f}", flush=True)
    print(f"[Multimodal Joint] Signal+Z={mae_j_both:.4f} | Signal-Only={mae_j_sig:.4f} | Z-Only={mae_j_tab:.4f}", flush=True)
    return out


if __name__ == "__main__":
    run_cwru_case_study()
    run_ecg_case_study()
    run_multimodal_ablation()
    run_low_data_curve()
