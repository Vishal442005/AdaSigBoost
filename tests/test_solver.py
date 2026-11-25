"""Unit tests for C++ coordinate descent solvers and histogram trees."""

from __future__ import annotations

import pickle

import numpy as np

from adasigboost import _adasigboost_cpp


def test_scalar_elastic_net_kkt_optimality() -> None:
    rng = np.random.default_rng(42)
    N, K = 150, 8
    Z = rng.normal(size=(N, K))
    h = rng.uniform(0.2, 2.0, size=N)
    w_true = np.array([1.5, -0.8, 0.0, 0.0, 2.0, 0.0, 0.0, -0.4])
    g = -h * (Z @ w_true + 0.1 * rng.normal(size=N))

    lambda1, lambda2 = 12.0, 1.5
    w_init = np.zeros(K, dtype=np.float64)
    w_opt, iters, max_delta = _adasigboost_cpp.solve_elastic_net_weights(
        Z, g, h, w_init, lambda1, lambda2, 1000, 1e-10
    )
    assert iters > 0
    assert max_delta < 1e-9

    # Verify exact KKT subdifferential conditions
    p = Z @ w_opt
    grad_quad = Z.T @ (g + h * p) + lambda2 * w_opt
    for k in range(K):
        if abs(w_opt[k]) > 1e-9:
            np.testing.assert_allclose(grad_quad[k] + lambda1 * np.sign(w_opt[k]), 0.0, atol=1e-6)
        else:
            assert abs(grad_quad[k]) <= lambda1 + 1e-6


def test_group_elastic_net_kkt_optimality() -> None:
    rng = np.random.default_rng(7)
    N, K, C = 120, 6, 4
    Z = rng.normal(size=(N, K, C))
    h = rng.uniform(0.1, 1.0, size=(N, C))
    W_true = np.zeros((K, C))
    W_true[0] = [1.2, -0.5, 0.3, -1.0]
    W_true[3] = [-0.8, 0.9, -0.1, 0.0]

    signal = np.einsum("nkc,kc->nc", Z, W_true)
    g = -h * (signal + 0.05 * rng.normal(size=(N, C)))

    lambda1, lambda2 = 8.0, 1.2
    W_init = np.zeros((K, C), dtype=np.float64)
    W_opt, _, max_delta = _adasigboost_cpp.solve_group_elastic_net_weights(
        Z, g, h, W_init, lambda1, lambda2, 1000, 1e-10
    )
    assert max_delta < 1e-8

    p = np.einsum("nkc,kc->nc", Z, W_opt)
    grad_quad = np.einsum("nkc,nc->kc", Z, g + h * p) + lambda2 * W_opt
    for k in range(K):
        wn = float(np.linalg.norm(W_opt[k]))
        gn = float(np.linalg.norm(grad_quad[k]))
        if wn > 1e-8:
            expected = -lambda1 * (W_opt[k] / wn)
            np.testing.assert_allclose(grad_quad[k], expected, atol=1e-5)
        else:
            assert gn <= lambda1 + 1e-5


def test_hist_tree_fit_predict_and_pickle() -> None:
    rng = np.random.default_rng(123)
    N, F = 400, 5
    X = rng.normal(size=(N, F))
    y = np.where(X[:, 0] > 0.0, 2.0, -2.0) + np.where(X[:, 2] > 0.5, 1.0, 0.0)
    g = -y
    h = np.ones(N, dtype=np.float64)

    X_binned, bin_edges = _adasigboost_cpp.compute_quantile_bins(X, 64)
    tree = _adasigboost_cpp.fit_hist_tree(X_binned, bin_edges, g, h, 3, 1.0, 0.0, 1e-3, 5)
    preds = tree.predict(X)
    assert np.corrcoef(preds, y)[0, 1] > 0.95

    raw_bytes = pickle.dumps(tree)
    tree_loaded = pickle.loads(raw_bytes)
    np.testing.assert_allclose(tree_loaded.predict(X), preds)
