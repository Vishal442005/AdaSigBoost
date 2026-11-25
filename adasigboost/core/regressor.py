"""Scikit-learn compatible AdaSigBoostRegressor."""

from __future__ import annotations

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d

from .base import AdaSigBoostBase
from .losses import HuberLoss, SquaredLoss


class AdaSigBoostRegressor(RegressorMixin, AdaSigBoostBase):
    """Adaptive Signal-Decomposed Second-Order Gradient Boosting Regressor."""

    def __init__(
        self,
        n_estimators: int = 50,
        n_modes: int = 8,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        loss: str = "squared_error",
        huber_delta: float = 1.35,
        lambda1: float = 0.5,
        lambda2: float = 1.0,
        alpha1: float = 0.05,
        alpha2: float = 0.01,
        reg_gamma: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_hessian: float = 1e-3,
        min_samples_leaf: int = 5,
        max_bins: int = 256,
        n_inner_iters: int = 2,
        n_filter_steps: int = 16,
        filter_lr: float = 0.08,
        dictionary_type: str = "composite",
        oof_folds: int = 5,
        use_oof: bool = True,
        mode_dropout: float = 0.0,
        use_bic_stopping: bool = True,
        bic_scale: float = 0.05,
        bic_patience: int = 5,
        min_stages: int = 5,
        freeze_decomposition: bool = False,
        uniform_weights: bool = False,
        global_tree_mode: bool = False,
        block1_mode: str = "full",
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            n_modes=n_modes,
            learning_rate=learning_rate,
            max_depth=max_depth,
            lambda1=lambda1,
            lambda2=lambda2,
            alpha1=alpha1,
            alpha2=alpha2,
            reg_gamma=reg_gamma,
            reg_lambda=reg_lambda,
            min_child_hessian=min_child_hessian,
            min_samples_leaf=min_samples_leaf,
            max_bins=max_bins,
            n_inner_iters=n_inner_iters,
            n_filter_steps=n_filter_steps,
            filter_lr=filter_lr,
            dictionary_type=dictionary_type,
            oof_folds=oof_folds,
            use_oof=use_oof,
            mode_dropout=mode_dropout,
            use_bic_stopping=use_bic_stopping,
            bic_scale=bic_scale,
            bic_patience=bic_patience,
            min_stages=min_stages,
            freeze_decomposition=freeze_decomposition,
            uniform_weights=uniform_weights,
            global_tree_mode=global_tree_mode,
            block1_mode=block1_mode,
            random_state=random_state,
        )
        self.loss = loss
        self.huber_delta = huber_delta

    def _make_loss(self, y: np.ndarray):
        if self.loss == "squared_error":
            return SquaredLoss()
        if self.loss == "huber":
            return HuberLoss(delta=self.huber_delta)
        raise ValueError(f"Unknown regression loss: {self.loss!r}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray | None = None,
    ) -> AdaSigBoostRegressor:
        """Fit AdaSigBoostRegressor on signal data X (and optional tabular covariates Z)."""
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None.")
        y_arr = check_array(y, ensure_2d=False, dtype=np.float64, accept_sparse=False)
        y_arr = column_or_1d(y_arr, warn=True)
        self._fit_core(X, y_arr, Z=Z)
        return self

    def predict(
        self,
        X: np.ndarray,
        Z: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict continuous targets for signal data X."""
        check_is_fitted(self, ["init_pred_", "stages_"])
        F, _ = self._raw_predict_and_contributions(X, Z=Z)
        return F[:, 0]

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Return R^2 coefficient of determination on (X, y) with optional covariates Z."""
        from sklearn.metrics import r2_score

        return float(r2_score(y, self.predict(X, Z=Z), sample_weight=sample_weight))
