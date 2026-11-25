"""Scikit-learn estimator compliance and multimodal/multiclass integration tests."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from adasigboost import AdaSigBoostClassifier, AdaSigBoostRegressor
from adasigboost.interpretability import compute_mode_deletion_auc


@parametrize_with_checks(
    [
        AdaSigBoostClassifier(
            n_estimators=4,
            n_modes=2,
            max_depth=2,
            n_filter_steps=2,
            lambda1=0.0,
            use_bic_stopping=False,
            random_state=0,
        ),
        AdaSigBoostRegressor(
            n_estimators=12,
            n_modes=2,
            max_depth=3,
            learning_rate=0.25,
            n_filter_steps=2,
            lambda1=0.0,
            use_bic_stopping=False,
            random_state=0,
        ),
    ]
)
def test_sklearn_estimator_compatibility(estimator, check) -> None:
    check(estimator)


def test_multichannel_multiclass_and_tabular_fusion() -> None:
    rng = np.random.default_rng(99)
    N, C, T = 240, 3, 128
    t = np.linspace(0.0, 1.0, T)
    y = rng.integers(0, 3, size=N)
    Z = rng.normal(size=(N, 4))

    X = 0.1 * rng.normal(size=(N, C, T))
    for i in range(N):
        if y[i] == 0:
            X[i, 0] += 1.5 * np.sin(2.0 * np.pi * 5.0 * t)
        elif y[i] == 1:
            X[i, 1] += 1.5 * np.sin(2.0 * np.pi * 18.0 * t)
        else:
            X[i, 2] += 1.5 * np.sin(2.0 * np.pi * 32.0 * t)
        # Tabular interaction boost
        if Z[i, 0] > 0.5 and y[i] == 2:
            X[i, 2] *= 1.3

    clf = AdaSigBoostClassifier(
        n_estimators=10,
        n_modes=6,
        max_depth=3,
        learning_rate=0.2,
        lambda1=0.5,
        use_bic_stopping=False,
        random_state=99,
    )
    clf.fit(X[:180], y[:180], Z=Z[:180])
    probs = clf.predict_proba(X[180:], Z=Z[180:])
    assert probs.shape == (60, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-6)
    assert clf.score(X[180:], y[180:], Z=Z[180:]) > 0.85

    contribs = clf.predict_mode_contributions(X[180:], Z=Z[180:])
    assert contribs.shape == (60, 6, 3)


def test_mode_deletion_faithfulness_auc() -> None:
    rng = np.random.default_rng(5)
    N, T = 200, 128
    t = np.linspace(0.0, 1.0, T)
    amp = rng.uniform(0.1, 2.0, size=N)
    X = amp[:, None] * np.sin(2.0 * np.pi * 12.0 * t)[None, :] + 0.1 * rng.normal(size=(N, T))
    y = (amp > 1.0).astype(int)

    clf = AdaSigBoostClassifier(
        n_estimators=8,
        n_modes=4,
        lambda1=0.2,
        use_bic_stopping=False,
        random_state=5,
    )
    clf.fit(X[:140], y[:140])
    res = compute_mode_deletion_auc(
        clf,
        X[140:],
        y[140:],
        metric_fn=lambda yt, F: roc_auc_score(yt, F[:, 0]),
    )
    assert res["deletion_scores"][0] > 0.90
    assert res["deletion_scores"][-1] <= 0.60
