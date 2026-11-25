"""Additional unit tests covering losses, BIC stopping, interpretability attribution, and plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pytest

from adasigboost import AdaSigBoostClassifier, AdaSigBoostRegressor
from adasigboost.core.losses import HuberLoss
from adasigboost.interpretability import (
    compute_mode_deletion_auc,
    extract_learned_frequencies,
    plot_critical_difference_approx,
    plot_mode_decomposition,
    plot_snr_benchmark,
    spectral_frequency_alignment_error,
    summarize_stage_modes,
)
from adasigboost.utils.validation import check_signal_and_tabular, ensure_3d_signals


def test_huber_loss_and_bic_global_dropout_branches(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    T = 128
    t = np.linspace(0.0, 1.0, T)
    N = 45
    a = rng.uniform(0.5, 2.0, size=N)
    Z = rng.normal(size=(N, 2))
    X = a[:, None] * np.sin(2.0 * np.pi * 8.0 * t)[None, :] + 0.05 * rng.normal(size=(N, T))
    y = 2.0 * a + 1.5 * Z[:, 0]

    loss = HuberLoss(delta=1.0)
    y_mat = np.zeros((N, 1))
    assert loss.init_prediction(y).shape == (1,)
    assert loss.loss(y, y_mat) > 0.0
    g, h = loss.grad_hess(y, y_mat)
    assert g.shape == (N, 1) and np.all(h > 0)

    # Fit with Huber loss + BIC early stopping + mode dropout + tabular Z
    reg_bic = AdaSigBoostRegressor(
        n_estimators=10,
        n_modes=4,
        loss="huber",
        use_bic_stopping=True,
        min_stages=2,
        bic_patience=1,
        bic_scale=50.0,
        mode_dropout=0.25,
        n_filter_steps=2,
        random_state=11,
    ).fit(X, y, Z=Z)
    assert 2 <= reg_bic.n_stages_fitted_ <= 10
    preds = reg_bic.predict(X, Z=Z)
    assert preds.shape == (N,)

    # Global tree mode branch (with and without Z)
    reg_glob = AdaSigBoostRegressor(
        n_estimators=3,
        n_modes=3,
        global_tree_mode=True,
        use_bic_stopping=False,
        n_filter_steps=2,
        random_state=11,
    ).fit(X, y, Z=Z)
    assert reg_glob.predict(X, Z=Z).shape == (N,)
    del_glob = compute_mode_deletion_auc(
        reg_glob,
        X,
        y,
        Z=Z,
        metric_fn=lambda yt, F: -float(np.mean((yt - F[:, 0]) ** 2)),
    )
    assert "deletion_auc" in del_glob

    # Interpretability & plotting helpers
    freqs = extract_learned_frequencies(reg_bic, fs=float(T))
    assert isinstance(freqs, list)
    align = spectral_frequency_alignment_error(reg_bic, [8.0, 16.0], fs=float(T))
    assert "mean_min_hz_error" in align
    summary = summarize_stage_modes(reg_bic, fs=float(T))
    assert len(summary) == 4

    fig1 = plot_mode_decomposition(reg_bic, X[0], fs=float(T), save_path=tmp_path / "m.pdf")
    fig2 = plot_snr_benchmark(
        [20.0, 10.0],
        {"AdaSigBoost": ([0.5, 1.0], [0.05, 0.1])},
        save_path=tmp_path / "snr.pdf",
    )
    fig3 = plot_critical_difference_approx(
        {"AdaSigBoost": 1.2, "LightGBM": 3.4},
        cd=1.1,
        save_path=tmp_path / "cd.pdf",
    )
    assert (tmp_path / "m.pdf").exists()
    assert (tmp_path / "snr.pdf").exists()
    assert (tmp_path / "cd.pdf").exists()
    assert fig1 is not None and fig2 is not None and fig3 is not None


def test_validation_and_error_guards() -> None:
    with pytest.raises(ValueError, match="Complex data"):
        ensure_3d_signals(np.ones((2, 1, 8), dtype=np.complex128))
    with pytest.raises(ValueError, match="NaN"):
        bad = np.ones((2, 2, 8), dtype=np.float64)
        bad[0, 0, 0] = np.nan
        ensure_3d_signals(bad)
    with pytest.raises(ValueError, match="Sample count mismatch"):
        check_signal_and_tabular(np.ones((4, 16)), Z=np.ones((3, 2)))

    X = np.random.default_rng(0).normal(size=(24, 2, 32))
    y = np.array([0, 1] * 12)
    Z = np.ones((24, 2))
    clf = AdaSigBoostClassifier(
        n_estimators=2, n_modes=2, n_filter_steps=2, use_bic_stopping=False, random_state=0
    ).fit(X, y, Z=Z)
    with pytest.raises(ValueError, match="Expected signal shape"):
        clf.predict(np.ones((4, 3, 32)), Z=np.ones((4, 2)))
    with pytest.raises(ValueError, match="Z must be provided"):
        clf.predict(X)
    with pytest.raises(ValueError, match="Expected Z with 2 features"):
        clf.predict(X, Z=np.ones((24, 3)))
