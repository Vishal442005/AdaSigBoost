"""Phase 5 Nested/Group/Temporal Cross-Validation Benchmark & Optuna Tuning Harness."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from adasigboost import AdaSigBoostClassifier, AdaSigBoostRegressor
from adasigboost.interpretability.plots import plot_critical_difference_approx
from benchmarks.baselines import (
    DeepTimeSeriesBaseline,
    LearnableFilterNeuralBaseline,
    RandomConvKernelBaseline,
    RawGBDTBaseline,
    StaticDecompGBDTBaseline,
)
from benchmarks.data_loaders import BenchmarkDataset, load_benchmark_dataset
from benchmarks.stats_analysis import (
    expected_calibration_error,
    friedman_nemenyi,
    holm_bonferroni_wilcoxon,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _score_predictions(
    ds: BenchmarkDataset,
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if ds.task == "regression":
        out["rmse"] = float(root_mean_squared_error(y_true, preds))
        out["mae"] = float(mean_absolute_error(y_true, preds))
        out["primary"] = out[ds.metric]
    elif ds.task == "binary":
        out["accuracy"] = float(accuracy_score(y_true, preds))
        if probs is not None and len(np.unique(y_true)) == 2:
            p1 = probs[:, 1] if probs.ndim == 2 else probs
            out["auroc"] = float(roc_auc_score(y_true, p1))
            out.update(expected_calibration_error(y_true, p1))
        out["primary"] = out.get(ds.metric, out["accuracy"])
    else:
        out["accuracy"] = float(accuracy_score(y_true, preds))
        if probs is not None and probs.shape[1] == len(np.unique(y_true)):
            out.update(expected_calibration_error(y_true, probs))
        out["primary"] = out["accuracy"]
    return out


def _get_splits(ds: BenchmarkDataset, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    N = len(ds.y)
    if ds.groups is not None:
        n_grp = len(np.unique(ds.groups))
        splitter = GroupKFold(n_splits=min(n_splits, n_grp))
        return list(splitter.split(np.zeros(N), ds.y, groups=ds.groups))
    if ds.domain == "Energy & Smart Grid":
        splits = []
        chunk = N // (n_splits + 1)
        for fold in range(n_splits):
            tr_end = (fold + 1) * chunk
            te_end = min(N, (fold + 2) * chunk)
            splits.append((np.arange(0, tr_end), np.arange(tr_end, te_end)))
        return splits
    if ds.task in {"binary", "multiclass"}:
        splitter_s = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter_s.split(np.zeros(N), ds.y))
    splitter_k = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter_k.split(np.zeros(N)))


def _build_model(model_name: str, ds: BenchmarkDataset, params: dict[str, Any], seed: int) -> Any:
    task_kind = "regression" if ds.task == "regression" else "classification"
    if model_name == "AdaSigBoost":
        cls = AdaSigBoostRegressor if task_kind == "regression" else AdaSigBoostClassifier
        return cls(
            n_estimators=params.get("n_estimators", 22),
            n_modes=params.get("n_modes", 6),
            learning_rate=params.get("learning_rate", 0.16),
            max_depth=params.get("max_depth", 4),
            lambda1=params.get("lambda1", 0.8),
            lambda2=params.get("lambda2", 1.0),
            alpha1=params.get("alpha1", 0.05),
            n_filter_steps=6,
            filter_lr=0.12,
            use_bic_stopping=False,
            random_state=seed,
        )
    if model_name == "XGBoost":
        return RawGBDTBaseline(
            backend="xgboost",
            task=task_kind,
            n_estimators=params.get("n_estimators", 120),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=seed,
        )
    if model_name == "LightGBM":
        return RawGBDTBaseline(
            backend="lightgbm",
            task=task_kind,
            n_estimators=params.get("n_estimators", 120),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=seed,
        )
    if model_name == "CatBoost":
        return RawGBDTBaseline(
            backend="catboost",
            task=task_kind,
            iterations=params.get("n_estimators", 120),
            depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            random_seed=seed,
        )
    if model_name == "DWT+LightGBM":
        return StaticDecompGBDTBaseline(
            decomp="dwt",
            gbdt="lightgbm",
            task=task_kind,
            n_modes=5,
            n_estimators=params.get("n_estimators", 120),
            max_depth=params.get("max_depth", 4),
            random_state=seed,
        )
    if model_name == "EMD+XGBoost":
        return StaticDecompGBDTBaseline(
            decomp="emd",
            gbdt="xgboost",
            task=task_kind,
            n_modes=4,
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 4),
            random_state=seed,
        )
    if model_name == "VMD+CatBoost":
        return StaticDecompGBDTBaseline(
            decomp="vmd",
            gbdt="catboost",
            task=task_kind,
            n_modes=5,
            iterations=params.get("n_estimators", 100),
            depth=params.get("max_depth", 4),
            random_seed=seed,
        )
    if model_name == "MultiRocket+Ridge":
        return RandomConvKernelBaseline(
            n_kernels=300,
            head="ridge",
            task=task_kind,
            seed=seed,
            alpha=params.get("alpha", 10.0),
        )
    if model_name == "MultiRocket+LightGBM":
        return RandomConvKernelBaseline(
            n_kernels=300,
            head="lightgbm",
            task=task_kind,
            seed=seed,
            n_estimators=100,
            max_depth=4,
            random_state=seed,
        )
    if model_name == "LearnableFilter+MLP":
        return LearnableFilterNeuralBaseline(
            n_filters=8,
            epochs=25,
            lr=params.get("lr", 1e-2),
            head="mlp",
            task=task_kind,
            seed=seed,
        )
    if model_name == "LearnableFilter+LightGBM":
        return LearnableFilterNeuralBaseline(
            n_filters=8,
            epochs=25,
            lr=params.get("lr", 1e-2),
            head="lightgbm",
            task=task_kind,
            seed=seed,
        )
    if model_name == "PatchTST":
        return DeepTimeSeriesBaseline(
            arch="patchtst",
            epochs=params.get("epochs", 20),
            batch_size=params.get("batch_size", 64),
            lr=params.get("lr", 3e-3),
            task=task_kind,
            seed=seed,
        )
    if model_name == "ModernTCN":
        return DeepTimeSeriesBaseline(
            arch="moderntcn",
            epochs=params.get("epochs", 20),
            batch_size=params.get("batch_size", 64),
            lr=params.get("lr", 3e-3),
            task=task_kind,
            seed=seed,
        )
    if model_name == "TimesNet":
        return DeepTimeSeriesBaseline(
            arch="timesnet",
            epochs=params.get("epochs", 20),
            batch_size=params.get("batch_size", 64),
            lr=params.get("lr", 3e-3),
            task=task_kind,
            seed=seed,
        )
    if model_name == "iTransformer":
        return DeepTimeSeriesBaseline(
            arch="itransformer",
            epochs=params.get("epochs", 20),
            batch_size=params.get("batch_size", 64),
            lr=params.get("lr", 3e-3),
            task=task_kind,
            seed=seed,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _plot_pareto_frontier(
    avg_ranks: dict[str, float],
    mean_times: dict[str, float],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for name, rank in avg_ranks.items():
        t_val = mean_times[name]
        color = "#2563eb" if name == "AdaSigBoost" else "#475569"
        marker = "*" if name == "AdaSigBoost" else "o"
        size = 180 if name == "AdaSigBoost" else 65
        ax.scatter(t_val, rank, c=color, marker=marker, s=size, zorder=4)
        ax.annotate(name, (t_val, rank), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Mean Training Time per Fold (s, log scale)")
    ax.set_ylabel("Average Rank across Datasets (Lower is Better)")
    ax.set_title("Accuracy-Rank vs. Training Compute Pareto Frontier")
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")


def run_benchmark(
    datasets: list[str],
    models: list[str],
    n_folds: int = 3,
    n_optuna_trials: int = 0,
    max_samples: int = 800,
    output_dir: Path = Path("results"),
    fig_dir: Path = Path("paper/figures"),
    seed: int = 42,
) -> dict[str, Any]:
    """Run cross-validated multi-dataset comparison and statistical significance tests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, Any]] = {}

    for ds_name in datasets:
        ds = load_benchmark_dataset(ds_name, seed=seed)
        if len(ds.y) > max_samples:
            rng = np.random.default_rng(seed)
            if ds.domain == "Energy & Smart Grid":
                idx = np.arange(max_samples)
            else:
                idx = np.sort(rng.choice(len(ds.y), size=max_samples, replace=False))
            ds.X = ds.X[idx]
            ds.y = ds.y[idx]
            if ds.Z is not None:
                ds.Z = ds.Z[idx]
            if ds.groups is not None:
                ds.groups = ds.groups[idx]

        splits = _get_splits(ds, n_splits=n_folds, seed=seed)
        all_results[ds_name] = {"metric": ds.metric, "task": ds.task, "models": {}}

        for m_name in models:
            fold_metrics = []
            train_times = []
            infer_latencies = []
            for tr_idx, te_idx in splits:
                X_tr, X_te = ds.X[tr_idx], ds.X[te_idx]
                y_tr, y_te = ds.y[tr_idx], ds.y[te_idx]
                Z_tr = ds.Z[tr_idx] if ds.Z is not None else None
                Z_te = ds.Z[te_idx] if ds.Z is not None else None

                best_params: dict[str, Any] = {}
                if n_optuna_trials > 0 and len(tr_idx) >= 30:
                    def _objective(
                        trial: optuna.Trial,
                        _m: str = m_name,
                        _ds: BenchmarkDataset = ds,
                        _xt: np.ndarray = X_tr,
                        _yt: np.ndarray = y_tr,
                        _zt: np.ndarray | None = Z_tr,
                    ) -> float:
                        p: dict[str, Any] = {}
                        if _m == "AdaSigBoost":
                            p = {
                                "n_modes": trial.suggest_categorical("n_modes", [4, 6, 8]),
                                "max_depth": trial.suggest_int("max_depth", 3, 5),
                                "lambda1": trial.suggest_float("lambda1", 0.05, 2.0, log=True),
                                "learning_rate": trial.suggest_float("learning_rate", 0.08, 0.25),
                            }
                        elif _m in {"XGBoost", "LightGBM", "CatBoost", "DWT+LightGBM", "VMD+CatBoost", "EMD+XGBoost"}:
                            p = {
                                "max_depth": trial.suggest_int("max_depth", 3, 6),
                                "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.25),
                            }
                        elif _m == "MultiRocket+Ridge":
                            p = {"alpha": trial.suggest_float("alpha", 0.1, 100.0, log=True)}
                        elif _m in {
                            "LearnableFilter+MLP",
                            "LearnableFilter+LightGBM",
                            "PatchTST",
                            "ModernTCN",
                            "TimesNet",
                            "iTransformer",
                        }:
                            p = {
                                "lr": trial.suggest_float("lr", 8e-4, 2e-2, log=True),
                                "epochs": trial.suggest_int("epochs", 12, 32, step=4),
                                "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
                            }

                        cut = max(10, int(0.8 * len(_yt)))
                        mdl = _build_model(_m, _ds, p, seed)
                        mdl.fit(_xt[:cut], _yt[:cut], Z=_zt[:cut] if _zt is not None else None)
                        pr = mdl.predict(_xt[cut:], Z=_zt[cut:] if _zt is not None else None)
                        pb = (
                            mdl.predict_proba(_xt[cut:], Z=_zt[cut:] if _zt is not None else None)
                            if (_ds.task != "regression" and hasattr(mdl, "predict_proba"))
                            else None
                        )
                        sc = _score_predictions(_ds, _yt[cut:], pr, pb)["primary"]
                        return -sc if _ds.task != "regression" else sc

                    study = optuna.create_study(direction="minimize")
                    study.optimize(_objective, n_trials=n_optuna_trials)
                    best_params = study.best_params

                model = _build_model(m_name, ds, best_params, seed)
                t0 = time.perf_counter()
                model.fit(X_tr, y_tr, Z=Z_tr)
                train_times.append(time.perf_counter() - t0)

                t_inf = time.perf_counter()
                preds = model.predict(X_te, Z=Z_te)
                infer_latencies.append(1000.0 * (time.perf_counter() - t_inf) / max(len(X_te), 1))

                probs = (
                    model.predict_proba(X_te, Z=Z_te)
                    if (ds.task != "regression" and hasattr(model, "predict_proba"))
                    else None
                )
                fold_metrics.append(_score_predictions(ds, y_te, preds, probs))

            primaries = [fm["primary"] for fm in fold_metrics]
            eces = [fm["ece"] for fm in fold_metrics if "ece" in fm]
            rec: dict[str, Any] = {
                "primary_mean": float(np.mean(primaries)),
                "primary_std": float(np.std(primaries)),
                "train_time_s": float(np.mean(train_times)),
                "infer_ms_per_sample": float(np.mean(infer_latencies)),
            }
            if eces:
                rec["ece_mean"] = float(np.mean(eces))
            all_results[ds_name]["models"][m_name] = rec
            print(
                f"[{ds_name:18s}] {m_name:24s} -> {ds.metric}={rec['primary_mean']:.4f} ± {rec['primary_std']:.4f} ({rec['train_time_s']:.2f}s)",
                flush=True,
            )

    if len(datasets) >= 2 and len(models) >= 3:
        score_mat = np.zeros((len(datasets), len(models)), dtype=np.float64)
        for i, d_name in enumerate(datasets):
            is_reg = all_results[d_name]["task"] == "regression"
            for j, m_name in enumerate(models):
                val = all_results[d_name]["models"][m_name]["primary_mean"]
                score_mat[i, j] = -val if is_reg else val
        fn = friedman_nemenyi(score_mat, models, higher_is_better=True)
        wb = (
            holm_bonferroni_wilcoxon(score_mat, models, reference_model="AdaSigBoost", higher_is_better=True)
            if "AdaSigBoost" in models
            else []
        )
        all_results["_stats"] = {"friedman_nemenyi": fn, "wilcoxon_holm": wb}
        plot_critical_difference_approx(
            fn["avg_ranks"],
            fn["critical_difference_05"],
            save_path=fig_dir / "fig_cd_diagram.pdf",
        )
        mean_times = {
            m: float(np.mean([all_results[d]["models"][m]["train_time_s"] for d in datasets]))
            for m in models
        }
        _plot_pareto_frontier(fn["avg_ranks"], mean_times, fig_dir / "fig_pareto.pdf")

    (output_dir / "real_benchmark.json").write_text(json.dumps(all_results, indent=2))
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "ECG5000",
            "eeg_eyestate",
            "Epilepsy",
            "cwru_bearing",
            "Wafer",
            "etth1_energy",
            "FordA",
            "InsectWingbeatSound",
        ],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "AdaSigBoost",
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "DWT+LightGBM",
            "VMD+CatBoost",
            "MultiRocket+Ridge",
            "MultiRocket+LightGBM",
            "LearnableFilter+MLP",
            "LearnableFilter+LightGBM",
            "PatchTST",
            "ModernTCN",
        ],
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=750)
    args = parser.parse_args()
    run_benchmark(
        datasets=args.datasets,
        models=args.models,
        n_folds=args.folds,
        n_optuna_trials=args.trials,
        max_samples=args.max_samples,
    )
