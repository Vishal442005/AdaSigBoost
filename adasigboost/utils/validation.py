"""Input validation and signal shape normalization utilities."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.utils.validation import check_array


def ensure_3d_signals(X: np.ndarray) -> np.ndarray:
    """Validate and coerce signal array to float64 3D shape (N, C, T)."""
    if sparse.issparse(X):
        check_array(X, accept_sparse=False)

    # Check if raw object is already 3D numeric without casting complex silently
    if isinstance(X, np.ndarray) and X.ndim == 3:
        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[0] < 1 or X_arr.shape[1] < 1 or X_arr.shape[2] < 1:
            raise ValueError(f"Signal dimensions must be positive, got shape {X_arr.shape}.")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Input contains NaN, infinity or a value too large for dtype('float64').")
        return X_arr

    X_2d = check_array(
        X,
        accept_sparse=False,
        allow_nd=True,
        dtype=np.float64,
        ensure_2d=True,
        ensure_min_features=1,
    )
    if X_2d.ndim == 2:
        return X_2d[:, np.newaxis, :]
    if X_2d.ndim == 3:
        return X_2d
    raise ValueError(f"Expected 2D (N, T) or 3D (N, C, T) signal array, got {X_2d.ndim}D.")


def check_signal_and_tabular(
    X: np.ndarray,
    Z: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate signal tensor X (N, C, T) and optional tabular covariates Z (N, P)."""
    X_3d = ensure_3d_signals(X)
    if Z is None:
        return X_3d, None
    Z_arr = check_array(Z, accept_sparse=False, allow_nd=False, dtype=np.float64, ensure_min_features=1)
    if Z_arr.shape[0] != X_3d.shape[0]:
        raise ValueError(
            f"Sample count mismatch between X ({X_3d.shape[0]}) and Z ({Z_arr.shape[0]})."
        )
    return X_3d, Z_arr
