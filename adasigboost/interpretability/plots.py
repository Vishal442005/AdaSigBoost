"""Publication-grade Matplotlib visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..decomposition.filterbank import apply_filterbank_np


def _apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_mode_decomposition(
    model: Any,
    x_sample: np.ndarray,
    fs: float = 1.0,
    stage_idx: int = 0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Visualize raw signal and top learned sub-signals at a given boosting stage."""
    _apply_pub_style()
    x_arr = np.asarray(x_sample, dtype=np.float64)
    if x_arr.ndim == 1:
        x_3d = x_arr[np.newaxis, np.newaxis, :]
    elif x_arr.ndim == 2:
        x_3d = x_arr[np.newaxis, :, :]
    else:
        x_3d = x_arr[:1]

    stage = model.stages_[stage_idx]
    X_fft = np.fft.rfft(x_3d, axis=-1)
    _, s_mix = apply_filterbank_np(X_fft, stage.filter_params)  # (1, K, T)
    T = x_3d.shape[-1]
    t = np.arange(T) / fs

    active = stage.active_modes or list(range(min(4, model.n_modes)))
    active = sorted(active, key=lambda k: float(np.linalg.norm(stage.weights[k])), reverse=True)[:4]

    n_rows = 1 + len(active)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 1.8 * n_rows), sharex=True)
    axes[0].plot(t, x_3d[0, 0], color="#1f2937", lw=1.1)
    axes[0].set_title("Raw Input Signal (Channel 0)")

    for row, k in enumerate(active, start=1):
        fp = stage.filter_params
        f_hz = fp.omega[k] * fs / (2.0 * np.pi)
        wn = float(np.linalg.norm(stage.weights[k]))
        axes[row].plot(t, s_mix[0, k], lw=1.1, color="#2563eb")
        axes[row].set_title(
            f"Mode {k} ({fp.atom_types[k]}) | f={f_hz:.2f} Hz | ||w||={wn:.3f}"
        )

    axes[-1].set_xlabel("Time (s)" if fs != 1.0 else "Sample Index")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_snr_benchmark(
    snr_levels: list[float],
    results_by_model: dict[str, tuple[list[float], list[float]]],
    ylabel: str = "Test RMSE",
    title: str = "Synthetic Benchmark Across Noise Levels (SNR)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot SNR degradation curves with mean +/- std error bands."""
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, (means, stds) in results_by_model.items():
        m = np.asarray(means)
        s = np.asarray(stds)
        lw = 2.4 if "AdaSigBoost" in name else 1.5
        marker = "o" if "AdaSigBoost" in name else "s"
        ax.plot(snr_levels, m, marker=marker, lw=lw, label=name)
        ax.fill_between(snr_levels, m - s, m + s, alpha=0.12)
    ax.invert_xaxis()
    ax.set_xlabel("Signal-to-Noise Ratio (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_critical_difference_approx(
    avg_ranks: dict[str, float],
    cd: float,
    title: str = "Average Ranks & Nemenyi Critical Difference (p < 0.05)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Render a clean horizontal rank / CD comparison chart."""
    _apply_pub_style()
    sorted_items = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    names = [k for k, _ in sorted_items]
    ranks = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(7.0, 0.45 * len(names) + 1.5))
    y_pos = np.arange(len(names))
    colors = ["#2563eb" if "AdaSigBoost" in n else "#64748b" for n in names]
    ax.barh(y_pos, ranks, color=colors, height=0.6)
    ax.errorbar(
        ranks[0],
        y_pos[0],
        xerr=[[0.0], [cd]],
        fmt="none",
        ecolor="#dc2626",
        elinewidth=2.0,
        capsize=4,
        label=f"CD = {cd:.2f}",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Average Rank (Lower is Better)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
