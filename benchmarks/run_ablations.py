"""Phase 6 Systematic Ablation Suite (ABL-1 through ABL-7)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error

from adasigboost import AdaSigBoostRegressor
from benchmarks.synthetic import generate_synthetic_benchmark


def run_ablation_suite(
    n_train: int = 800,
    n_test: int = 400,
    signal_length: int = 512,
    seeds: list[int] | None = None,
    output_dir: Path = Path("results"),
) -> dict[str, Any]:
    """Run Full AdaSigBoost vs. ABL-1 through ABL-7 on low-SNR (+5 dB) and clean (+20 dB) tasks."""
    if seeds is None:
        seeds = [0, 1]
    output_dir.mkdir(parents=True, exist_ok=True)

    variants: dict[str, dict[str, Any]] = {
        "Full AdaSigBoost": {},
        "ABL-1 (Open-Loop Freeze)": {"freeze_decomposition": True},
        "ABL-2 (Uniform Weights)": {"uniform_weights": True},
        "ABL-3 (Global Shared Tree)": {"global_tree_mode": True},
        "ABL-4 (No Sparsity λ1=0)": {"lambda1": 0.0},
        "ABL-5 (No Orthogonality α1=0)": {"alpha1": 0.0},
        "ABL-6 (In-Sample Residuals)": {"use_oof": False},
        "ABL-7 (Energy-Only Block 1)": {"block1_mode": "energy_only"},
    }

    results: dict[str, dict[str, Any]] = {}
    for snr in [20.0, 5.0]:
        snr_key = f"snr_{int(snr)}db"
        results[snr_key] = {}
        for v_name, overrides in variants.items():
            rmses, r2s, active_counts, times = [], [], [], []
            for seed in seeds:
                ds = generate_synthetic_benchmark(
                    n_samples=n_train + n_test,
                    signal_length=signal_length,
                    snr_db=snr,
                    seed=seed,
                )
                kwargs: dict[str, Any] = {
                    "n_estimators": 22,
                    "n_modes": 8,
                    "learning_rate": 0.18,
                    "max_depth": 4,
                    "lambda1": 15.0,
                    "lambda2": 1.0,
                    "alpha1": 0.05,
                    "n_filter_steps": 6,
                    "filter_lr": 0.12,
                    "use_bic_stopping": False,
                    "random_state": seed,
                }
                kwargs.update(overrides)
                model = AdaSigBoostRegressor(**kwargs)

                t0 = time.perf_counter()
                model.fit(ds.X[:n_train], ds.y[:n_train])
                times.append(time.perf_counter() - t0)

                preds = model.predict(ds.X[n_train:])
                rmses.append(float(root_mean_squared_error(ds.y[n_train:], preds)))
                r2s.append(float(r2_score(ds.y[n_train:], preds)))
                active_counts.append(float(np.mean(model.active_mode_counts_)))

            summary = {
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "r2_mean": float(np.mean(r2s)),
                "avg_active_modes": float(np.mean(active_counts)),
                "train_time_s": float(np.mean(times)),
            }
            results[snr_key][v_name] = summary
            print(
                f"[{snr_key}] {v_name:28s} -> RMSE={summary['rmse_mean']:.4f}, R2={summary['r2_mean']:.4f}, active_K={summary['avg_active_modes']:.2f}",
                flush=True,
            )

    (output_dir / "ablations.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_ablation_suite()
