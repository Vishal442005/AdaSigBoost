"""Phase 4 Controlled Synthetic Benchmark Runner (SNR curves, mode recovery, K-sensitivity)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error

from adasigboost import AdaSigBoostRegressor
from adasigboost.decomposition.filterbank import apply_filterbank_np
from adasigboost.interpretability.plots import plot_mode_decomposition, plot_snr_benchmark
from benchmarks.baselines import (
    LearnableFilterNeuralBaseline,
    RandomConvKernelBaseline,
    RawGBDTBaseline,
    StaticDecompGBDTBaseline,
)
from benchmarks.synthetic import compute_spectral_iou, generate_synthetic_benchmark


def _evaluate_model(model: Any, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray) -> dict[str, float]:
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = model.predict(X_te)
    infer_ms = 1000.0 * (time.perf_counter() - t1) / max(len(X_te), 1)

    return {
        "rmse": float(root_mean_squared_error(y_te, preds)),
        "r2": float(r2_score(y_te, preds)),
        "train_time_s": float(train_s),
        "infer_ms_per_sample": float(infer_ms),
    }


def run_synthetic_suite(
    n_train: int = 500,
    n_test: int = 250,
    signal_length: int = 512,
    snr_levels: list[float] | None = None,
    seeds: list[int] | None = None,
    output_dir: Path = Path("results"),
    fig_dir: Path = Path("paper/figures"),
) -> dict[str, Any]:
    """Execute SNR sweep, mode recovery IoU, and K-sensitivity benchmarks."""
    if snr_levels is None:
        snr_levels = [20.0, 10.0, 5.0, 0.0, -5.0]
    if seeds is None:
        seeds = [0, 1]

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_names = [
        "AdaSigBoost",
        "XGBoost (Raw)",
        "LightGBM (Raw)",
        "DWT+LightGBM",
        "MultiRocket+Ridge",
        "LearnableFilter+MLP",
    ]
    snr_records: dict[str, dict[float, list[dict[str, float]]]] = {
        m: {snr: [] for snr in snr_levels} for m in model_names
    }

    for snr in snr_levels:
        for seed in seeds:
            ds = generate_synthetic_benchmark(
                n_samples=n_train + n_test,
                signal_length=signal_length,
                snr_db=snr,
                seed=seed,
            )
            X_tr, X_te = ds.X[:n_train], ds.X[n_train:]
            y_tr, y_te = ds.y[:n_train], ds.y[n_train:]

            candidates = {
                "AdaSigBoost": AdaSigBoostRegressor(
                    n_estimators=22,
                    n_modes=8,
                    learning_rate=0.18,
                    max_depth=4,
                    lambda1=1.0,
                    lambda2=1.0,
                    n_filter_steps=6,
                    filter_lr=0.12,
                    use_bic_stopping=False,
                    random_state=seed,
                ),
                "XGBoost (Raw)": RawGBDTBaseline(
                    backend="xgboost", task="regression", n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed
                ),
                "LightGBM (Raw)": RawGBDTBaseline(
                    backend="lightgbm", task="regression", n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed
                ),
                "DWT+LightGBM": StaticDecompGBDTBaseline(
                    decomp="dwt", gbdt="lightgbm", task="regression", n_modes=5, n_estimators=100, max_depth=4, random_state=seed
                ),
                "MultiRocket+Ridge": RandomConvKernelBaseline(
                    n_kernels=250, head="ridge", task="regression", seed=seed, alpha=10.0
                ),
                "LearnableFilter+MLP": LearnableFilterNeuralBaseline(
                    n_filters=8, epochs=30, lr=1e-2, task="regression", seed=seed
                ),
            }

            for m_name, clf in candidates.items():
                res = _evaluate_model(clf, X_tr, y_tr, X_te, y_te)
                snr_records[m_name][snr].append(res)
                print(
                    f"[SNR={snr:+5.1f} dB | seed={seed}] {m_name:22s} -> RMSE={res['rmse']:.4f}, R2={res['r2']:.4f}, t={res['train_time_s']:.2f}s",
                    flush=True,
                )

    # Aggregate SNR plot data
    plot_data: dict[str, tuple[list[float], list[float]]] = {}
    summary_snr: dict[str, dict[str, dict[str, float]]] = {}
    for m_name in model_names:
        means, stds = [], []
        summary_snr[m_name] = {}
        for snr in snr_levels:
            rmses = [r["rmse"] for r in snr_records[m_name][snr]]
            r2s = [r["r2"] for r in snr_records[m_name][snr]]
            times = [r["train_time_s"] for r in snr_records[m_name][snr]]
            means.append(float(np.mean(rmses)))
            stds.append(float(np.std(rmses)))
            summary_snr[m_name][str(snr)] = {
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "r2_mean": float(np.mean(r2s)),
                "train_time_s": float(np.mean(times)),
            }
        plot_data[m_name] = (means, stds)

    plot_snr_benchmark(
        snr_levels,
        plot_data,
        ylabel="Test RMSE (Lower is Better)",
        save_path=fig_dir / "fig_snr_curve.pdf",
    )

    # Mode recovery IoU and decomposition visualization at +20 dB
    ds_clean = generate_synthetic_benchmark(n_samples=n_train + n_test, signal_length=signal_length, snr_db=20.0, seed=0)
    asb_clean = AdaSigBoostRegressor(
        n_estimators=25,
        n_modes=8,
        learning_rate=0.15,
        max_depth=4,
        lambda1=1.0,
        n_filter_steps=12,
        use_bic_stopping=False,
        random_state=0,
    ).fit(ds_clean.X[:n_train], ds_clean.y[:n_train])

    X_te_fft = np.fft.rfft(ds_clean.X[n_train:, np.newaxis, :], axis=-1)
    ious = {}
    for idx, cname in enumerate(["trend", "chirp_threshold", "transient_morlet"]):
        true_c = ds_clean.components[n_train:, idx, :]
        best_iou = 0.0
        for st in asb_clean.stages_:
            _, s_mix_te = apply_filterbank_np(X_te_fft, st.filter_params)
            for k in st.active_modes:
                best_iou = max(best_iou, compute_spectral_iou(s_mix_te[:, k, :], true_c))
        ious[cname] = float(best_iou)

    plot_mode_decomposition(
        asb_clean,
        ds_clean.X[n_train],
        fs=ds_clean.fs,
        stage_idx=0,
        save_path=fig_dir / "fig_synthetic_modes.pdf",
    )

    # K-sensitivity curve at SNR = +10 dB
    ds_k = generate_synthetic_benchmark(n_samples=n_train + n_test, signal_length=signal_length, snr_db=10.0, seed=0)
    k_values = [2, 4, 8, 12, 16]
    k_sens = {}
    for k_val in k_values:
        m_k = AdaSigBoostRegressor(
            n_estimators=20,
            n_modes=k_val,
            learning_rate=0.15,
            max_depth=4,
            lambda1=12.0,
            n_filter_steps=6,
            use_bic_stopping=False,
            random_state=0,
        )
        res_k = _evaluate_model(m_k, ds_k.X[:n_train], ds_k.y[:n_train], ds_k.X[n_train:], ds_k.y[n_train:])
        res_k["avg_active_modes"] = float(np.mean(m_k.active_mode_counts_))
        k_sens[str(k_val)] = res_k

    # Plot K-sensitivity figure
    import matplotlib.pyplot as plt

    fig_k, ax1 = plt.subplots(figsize=(6.0, 3.8))
    rmses_k = [k_sens[str(k)]["rmse"] for k in k_values]
    acts_k = [k_sens[str(k)]["avg_active_modes"] for k in k_values]
    ax1.plot(k_values, rmses_k, "o-", color="#2563eb", lw=2.2, label="Test RMSE (Left)")
    ax1.set_xlabel("Candidate Modes K")
    ax1.set_ylabel("Test RMSE (+10 dB)", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(k_values, acts_k, "s--", color="#dc2626", lw=1.8, label="Avg Active Modes (Right)")
    ax2.set_ylabel("Avg Active Modes after L1 Pruning", color="#dc2626")
    ax1.set_title("Sensitivity to Candidate Mode Count K (L1 Pruning Stability)")
    ax1.grid(alpha=0.25, linestyle="--")
    fig_k.tight_layout()
    fig_k.savefig(fig_dir / "fig_k_sensitivity.pdf", bbox_inches="tight")

    # Scaling curve vs N and T (Eq. 5 verification)
    scaling: dict[str, list[dict[str, float]]] = {"by_N": [], "by_T": []}
    for n_sz in [200, 400, 800]:
        ds_s = generate_synthetic_benchmark(n_samples=n_sz, signal_length=512, snr_db=10.0, seed=0)
        t0 = time.perf_counter()
        AdaSigBoostRegressor(n_estimators=10, n_modes=6, n_filter_steps=5, use_bic_stopping=False, random_state=0).fit(ds_s.X, ds_s.y)
        scaling["by_N"].append({"N": float(n_sz), "time_s": float(time.perf_counter() - t0)})
    for t_len in [256, 512, 1024]:
        ds_s = generate_synthetic_benchmark(n_samples=400, signal_length=t_len, snr_db=10.0, seed=0)
        t0 = time.perf_counter()
        AdaSigBoostRegressor(n_estimators=10, n_modes=6, n_filter_steps=5, use_bic_stopping=False, random_state=0).fit(ds_s.X, ds_s.y)
        scaling["by_T"].append({"T": float(t_len), "time_s": float(time.perf_counter() - t0)})

    fig_sc, (ax_n, ax_t) = plt.subplots(1, 2, figsize=(7.6, 3.3))
    ax_n.plot([r["N"] for r in scaling["by_N"]], [r["time_s"] for r in scaling["by_N"]], "o-", color="#2563eb", lw=2)
    ax_n.set_xlabel("Sample Size N")
    ax_n.set_ylabel("Train Time (s)")
    ax_n.set_title("Linear Scaling in N")
    ax_n.grid(alpha=0.25, linestyle="--")
    ax_t.plot([r["T"] for r in scaling["by_T"]], [r["time_s"] for r in scaling["by_T"]], "s-", color="#0f766e", lw=2)
    ax_t.set_xlabel("Signal Length T")
    ax_t.set_ylabel("Train Time (s)")
    ax_t.set_title("Quasi-Linear O(T log T) Scaling")
    ax_t.grid(alpha=0.25, linestyle="--")
    fig_sc.tight_layout()
    fig_sc.savefig(fig_dir / "fig_scaling.pdf", bbox_inches="tight")

    payload = {
        "snr_summary": summary_snr,
        "spectral_iou_20db": ious,
        "k_sensitivity_10db": k_sens,
        "scaling": scaling,
    }
    (output_dir / "synthetic_suite.json").write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run_synthetic_suite()
