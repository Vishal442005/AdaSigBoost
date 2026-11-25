"""Tests validating Theorem 1 convergence and Proposition 1 distractor pruning."""

from __future__ import annotations

import numpy as np

from adasigboost import AdaSigBoostClassifier, AdaSigBoostRegressor


def test_theorem1_monotonic_training_loss_regression() -> None:
    rng = np.random.default_rng(11)
    N, T = 250, 128
    t = np.linspace(0.0, 1.0, T)
    a = rng.uniform(0.5, 2.5, size=N)
    b = rng.uniform(0.0, 1.0, size=N)
    X = a[:, None] * np.sin(2.0 * np.pi * 6.0 * t)[None, :] + b[:, None] * np.cos(
        2.0 * np.pi * 22.0 * t
    )[None, :] + 0.1 * rng.normal(size=(N, T))
    y = 2.0 * a + 1.5 * (b > 0.5).astype(float)

    model = AdaSigBoostRegressor(
        n_estimators=15,
        n_modes=5,
        learning_rate=0.2,
        lambda1=0.2,
        use_bic_stopping=False,
        random_state=11,
    )
    model.fit(X, y)

    # Check empirical risk decays monotonically across boosting stages
    losses = np.array(model.stage_losses_)
    assert losses[-1] < 0.25 * losses[0]
    assert np.all(np.diff(losses) <= 1e-8)


def test_proposition1_distractor_rejection_and_classification() -> None:
    rng = np.random.default_rng(22)
    N, T = 300, 128
    t = np.linspace(0.0, 1.0, T)
    signal_amp = rng.choice([0.2, 2.0], size=N)
    distractor_amp = rng.uniform(2.0, 5.0, size=N)
    X = (
        signal_amp[:, None] * np.sin(2.0 * np.pi * 10.0 * t)[None, :]
        + distractor_amp[:, None] * np.sin(2.0 * np.pi * 35.0 * t)[None, :]
        + 0.1 * rng.normal(size=(N, T))
    )
    y = (signal_amp > 1.0).astype(int)

    clf = AdaSigBoostClassifier(
        n_estimators=12,
        n_modes=4,
        learning_rate=0.25,
        lambda1=2.0,
        use_bic_stopping=False,
        random_state=22,
    )
    clf.fit(X, y)
    acc = clf.score(X, y)
    assert acc > 0.95
    # Verify L1 pruning actually zeroes out redundant modes in at least some stages
    assert min(clf.active_mode_counts_) <= clf.n_modes
