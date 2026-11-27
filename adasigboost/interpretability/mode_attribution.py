"""Quantitative faithfulness and physical frequency alignment metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_learned_frequencies(model: Any, fs: float = 1.0) -> list[dict[str, Any]]:
    """Extract learned center frequencies (in Hz if fs is sampling rate) and weights per stage."""
    records: list[dict[str, Any]] = []
    for m, stage in enumerate(model.stages_):
        fp = stage.filter_params
        freqs_hz = fp.omega * (fs / (2.0 * np.pi))
        for k in stage.active_modes:
            w_norm = float(np.linalg.norm(stage.weights[k]))
            records.append(
                {
                    "stage": m,
                    "mode": k,
                    "atom_type": fp.atom_types[k],
                    "omega_rad": float(fp.omega[k]),
                    "freq_hz": float(freqs_hz[k]),
                    "bandwidth_rad": float(fp.sigma[k]),
                    "weight_norm": w_norm,
                }
            )
    return records


def spectral_frequency_alignment_error(
    model: Any,
    target_freqs_hz: list[float] | np.ndarray,
    fs: float,
) -> dict[str, Any]:
    """Compute minimum Hz error between learned unique active mode frequencies and physical target Hz."""
    summary = summarize_stage_modes(model, fs=fs)
    active = [
        (v["cum_weight_norm"], v["weighted_freq_hz"])
        for v in summary.values()
        if v["cum_weight_norm"] > 0.0
    ]
    if not active:
        records = extract_learned_frequencies(model, fs=fs)
        if not records:
            return {"mean_min_hz_error": float("inf"), "max_min_hz_error": float("inf")}
        active = [(r["weight_norm"], r["freq_hz"]) for r in records]

    active.sort(key=lambda pair: pair[0], reverse=True)
    top_freqs = np.array([f for _, f in active], dtype=np.float64)

    errors = [float(np.min(np.abs(top_freqs - f_true))) for f_true in target_freqs_hz]
    matched = [float(top_freqs[int(np.argmin(np.abs(top_freqs - f_true)))]) for f_true in target_freqs_hz]
    return {
        "mean_min_hz_error": float(np.mean(errors)),
        "max_min_hz_error": float(np.max(errors)),
        "matched_freqs_hz": matched,
    }


def _mode_base_margin(model: Any, X: np.ndarray, F_full: np.ndarray, contribs: np.ndarray) -> np.ndarray:
    if getattr(model, "n_tabular_features_", 0) == 0:
        return np.tile(model.init_pred_[np.newaxis, :], (len(X), 1))
    return F_full - np.sum(contribs, axis=1)


def compute_mode_deletion_auc(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric_fn: Any,
    Z: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute Mode Deletion Faithfulness AUC by masking modes ranked by importance."""
    F_full, contribs = model._raw_predict_and_contributions(X, Z=Z)
    K = contribs.shape[1]
    mode_rank = np.argsort(np.sum(np.abs(contribs), axis=(0, 2)))[::-1]

    base_margin = _mode_base_margin(model, X, F_full, contribs)
    scores = [float(metric_fn(y, F_full))]
    kept = set(range(K))
    for k in mode_rank:
        kept.remove(int(k))
        if kept:
            F_curr = base_margin + np.sum(contribs[:, sorted(kept), :], axis=1)
        else:
            F_curr = base_margin.copy()
        scores.append(float(metric_fn(y, F_curr)))

    x_grid = np.linspace(0.0, 1.0, len(scores))
    auc = float(np.trapezoid(scores, x_grid)) if hasattr(np, "trapezoid") else float(np.trapz(scores, x_grid))
    return {
        "deletion_scores": scores,
        "deletion_auc": auc,
        "mode_order": [int(k) for k in mode_rank],
        "n_modes": K,
    }


def compute_mode_insertion_auc(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric_fn: Any,
    Z: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute Mode Insertion Faithfulness AUC by adding top-ranked modes from base margin."""
    F_full, contribs = model._raw_predict_and_contributions(X, Z=Z)
    K = contribs.shape[1]
    mode_rank = np.argsort(np.sum(np.abs(contribs), axis=(0, 2)))[::-1]

    base_margin = _mode_base_margin(model, X, F_full, contribs)
    F_curr = base_margin.copy()
    scores = [float(metric_fn(y, F_curr))]
    for k in mode_rank:
        F_curr = F_curr + contribs[:, int(k), :]
        scores.append(float(metric_fn(y, F_curr)))

    x_grid = np.linspace(0.0, 1.0, len(scores))
    auc = float(np.trapezoid(scores, x_grid)) if hasattr(np, "trapezoid") else float(np.trapz(scores, x_grid))
    return {
        "insertion_scores": scores,
        "insertion_auc": auc,
        "mode_order": [int(k) for k in mode_rank],
        "n_modes": K,
    }


def summarize_stage_modes(model: Any, fs: float = 1.0) -> dict[int, dict[str, Any]]:
    """Aggregate cumulative weight and dominant frequency per mode index k=0..K-1."""
    summary: dict[int, dict[str, Any]] = {}
    for k in range(model.n_modes):
        summary[k] = {"cum_weight_norm": 0.0, "weighted_freq_hz": 0.0, "active_stages": 0}
    for stage in model.stages_:
        fp = stage.filter_params
        freqs_hz = fp.omega * (fs / (2.0 * np.pi))
        for k in stage.active_modes:
            wn = float(np.linalg.norm(stage.weights[k]))
            summary[k]["cum_weight_norm"] += wn
            summary[k]["weighted_freq_hz"] += wn * float(freqs_hz[k])
            summary[k]["active_stages"] += 1
    for k in range(model.n_modes):
        tot = summary[k]["cum_weight_norm"]
        if tot > 0:
            summary[k]["weighted_freq_hz"] /= tot
    return summary
