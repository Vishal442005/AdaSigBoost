"""Non-parametric statistical tests (Friedman, Nemenyi CD, Holm-Bonferroni Wilcoxon) & calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

# Critical values q_{0.05} for two-tailed Nemenyi test (Demsar 2006) for k=2..15 algorithms
_NEMENYI_Q05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
}


def friedman_nemenyi(
    scores_matrix: np.ndarray,
    model_names: list[str],
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Run Friedman test and Nemenyi post-hoc Critical Difference across datasets x models.

    scores_matrix : shape (n_datasets, n_models)
    """
    n_datasets, n_models = scores_matrix.shape
    if n_models < 2 or n_datasets < 2:
        raise ValueError("Need at least 2 datasets and 2 models for Friedman test.")

    vals = -scores_matrix if higher_is_better else scores_matrix
    ranks = np.array([stats.rankdata(row) for row in vals], dtype=np.float64)
    avg_ranks = np.mean(ranks, axis=0)

    stat, p_val = stats.friedmanchisquare(*[scores_matrix[:, j] for j in range(n_models)])
    q_alpha = _NEMENYI_Q05.get(n_models, 3.391)
    cd = float(q_alpha * np.sqrt(n_models * (n_models + 1.0) / (6.0 * n_datasets)))

    return {
        "friedman_stat": float(stat),
        "p_value": float(p_val),
        "critical_difference_05": cd,
        "avg_ranks": {name: float(r) for name, r in zip(model_names, avg_ranks, strict=True)},
    }


def holm_bonferroni_wilcoxon(
    scores_matrix: np.ndarray,
    model_names: list[str],
    reference_model: str = "AdaSigBoost",
    higher_is_better: bool = True,
) -> list[dict[str, Any]]:
    """Pairwise Wilcoxon signed-rank tests vs. reference_model with Holm-Bonferroni correction."""
    ref_idx = model_names.index(reference_model)
    ref_scores = scores_matrix[:, ref_idx]
    raw_results = []

    for j, name in enumerate(model_names):
        if j == ref_idx:
            continue
        diff = (ref_scores - scores_matrix[:, j]) if higher_is_better else (scores_matrix[:, j] - ref_scores)
        if np.allclose(diff, 0.0):
            p_val = 1.0
        else:
            _, p_val = stats.wilcoxon(diff, alternative="greater", zero_method="zsplit")
        raw_results.append(
            {
                "baseline": name,
                "mean_diff": float(np.mean(diff)),
                "p_raw": float(p_val),
            }
        )

    # Holm-Bonferroni step-down adjustment
    raw_results.sort(key=lambda r: r["p_raw"])
    m = len(raw_results)
    prev_adj = 0.0
    for idx, r in enumerate(raw_results):
        adj = min(1.0, max(prev_adj, r["p_raw"] * (m - idx)))
        r["p_holm"] = float(adj)
        r["significant_05"] = bool(adj < 0.05)
        prev_adj = adj
    return raw_results


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute Expected Calibration Error (ECE) and Brier Score for classification."""
    if y_prob.ndim == 2:
        conf = np.max(y_prob, axis=1)
        pred = np.argmax(y_prob, axis=1)
        acc = (pred == y_true).astype(np.float64)
        # Multi-class Brier score
        onehot = np.zeros_like(y_prob)
        onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
        brier = float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))
    else:
        conf = np.maximum(y_prob, 1.0 - y_prob)
        pred = (y_prob >= 0.5).astype(int)
        acc = (pred == y_true).astype(np.float64)
        brier = float(np.mean((y_prob - y_true) ** 2))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = float(len(y_true))
    for b in range(n_bins):
        mask = (conf > bins[b]) & (conf <= bins[b + 1]) if b > 0 else (conf >= bins[b]) & (conf <= bins[b + 1])
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / N) * abs(float(np.mean(acc[mask])) - float(np.mean(conf[mask])))
    return {"ece": float(ece), "brier": brier}
