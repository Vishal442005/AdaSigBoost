"""Unit tests for synthetic generator and statistical significance tests."""

from __future__ import annotations

import numpy as np

from benchmarks.stats_analysis import (
    expected_calibration_error,
    friedman_nemenyi,
    holm_bonferroni_wilcoxon,
)
from benchmarks.synthetic import compute_spectral_iou, generate_synthetic_benchmark


def test_synthetic_generator_shapes_and_snr() -> None:
    ds_hi = generate_synthetic_benchmark(n_samples=64, signal_length=256, snr_db=20.0, seed=0)
    ds_lo = generate_synthetic_benchmark(n_samples=64, signal_length=256, snr_db=-5.0, seed=0)
    assert ds_hi.X.shape == (64, 256)
    assert ds_hi.components.shape == (64, 5, 256)
    assert np.var(ds_lo.X) > np.var(ds_hi.X)
    # Identical waveform has IoU == 1.0
    assert abs(compute_spectral_iou(ds_hi.components[:, 2, :], ds_hi.components[:, 2, :]) - 1.0) < 1e-6


def test_friedman_nemenyi_and_wilcoxon_holm() -> None:
    # 8 datasets x 4 models where model 0 clearly wins every dataset
    scores = np.array(
        [
            [0.95, 0.90, 0.88, 0.85],
            [0.92, 0.89, 0.87, 0.84],
            [0.89, 0.85, 0.83, 0.80],
            [0.96, 0.91, 0.89, 0.86],
            [0.91, 0.87, 0.85, 0.82],
            [0.94, 0.88, 0.86, 0.81],
            [0.90, 0.86, 0.84, 0.79],
            [0.93, 0.89, 0.87, 0.83],
        ]
    )
    names = ["AdaSigBoost", "VMD+GBDT", "MultiRocket", "XGBoost"]
    fn = friedman_nemenyi(scores, names, higher_is_better=True)
    assert fn["p_value"] < 0.001
    assert fn["avg_ranks"]["AdaSigBoost"] == 1.0

    wb = holm_bonferroni_wilcoxon(scores, names, reference_model="AdaSigBoost", higher_is_better=True)
    assert all(r["significant_05"] for r in wb)


def test_expected_calibration_error() -> None:
    y = np.array([0, 0, 1, 1])
    p_perfect = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    cal = expected_calibration_error(y, p_perfect)
    assert cal["ece"] < 1e-9
    assert cal["brier"] < 1e-9
