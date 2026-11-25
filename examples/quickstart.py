"""Minimal Quickstart example demonstrating AdaSigBoost on multi-channel + tabular signals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from adasigboost import AdaSigBoostClassifier
from adasigboost.interpretability import compute_mode_deletion_auc, extract_learned_frequencies


def main() -> None:
    rng = np.random.default_rng(42)
    N, C, T, fs = 240, 2, 256, 256.0
    t = np.linspace(0.0, 1.0, T, endpoint=False)

    # Synthetic multi-channel signal with localized 35 Hz transient + low-freq baseline + distractor
    y = rng.integers(0, 2, size=N)
    Z = rng.normal(size=(N, 3))
    X = 0.25 * rng.normal(size=(N, C, T))
    for i in range(N):
        # Distractor oscillation (70 Hz) orthogonal to label
        X[i, 0] += 1.2 * np.sin(2.0 * np.pi * 70.0 * t + rng.uniform(0, 2 * np.pi))
        # Relevant signal event on channel 0 when y == 1 + tabular interaction
        if y[i] == 1:
            t0 = rng.uniform(0.2, 0.8)
            env = np.exp(-0.5 * ((t - t0) / 0.05) ** 2)
            X[i, 0] += 0.9 * env * np.sin(2.0 * np.pi * 35.0 * (t - t0))
        X[i, 1] += 0.4 * Z[i, 0] * np.cos(2.0 * np.pi * 4.0 * t)

    X_tr, X_te, Z_tr, Z_te, y_tr, y_te = train_test_split(
        X, Z, y, test_size=0.3, stratify=y, random_state=42
    )

    clf = AdaSigBoostClassifier(
        n_estimators=20,
        n_modes=6,
        learning_rate=0.16,
        max_depth=3,
        lambda1=1.0,
        n_filter_steps=6,
        random_state=42,
    )
    clf.fit(X_tr, y_tr, Z=Z_tr)
    preds = clf.predict(X_te, Z=Z_te)
    print(f"Test Accuracy: {accuracy_score(y_te, preds):.4f}")

    contribs = clf.predict_mode_contributions(X_te, Z=Z_te)
    print(f"Per-mode contributions shape: {contribs.shape}")

    top_freqs = sorted(extract_learned_frequencies(clf, fs=fs), key=lambda r: r["weight_norm"], reverse=True)
    for rec in top_freqs[:4]:
        print(
            f"  Stage {rec['stage']:2d} | Mode {rec['mode']} ({rec['atom_type']:8s}) "
            f"| f={rec['freq_hz']:5.1f} Hz | ||w||={rec['weight_norm']:.3f}"
        )

    del_auc = compute_mode_deletion_auc(
        clf, X_te, y_te, Z=Z_te, metric_fn=lambda yt, F: accuracy_score(yt, (F[:, 0] > 0).astype(int))
    )
    print(f"Mode Deletion Faithfulness AUC: {del_auc['deletion_auc']:.4f}")


if __name__ == "__main__":
    main()
