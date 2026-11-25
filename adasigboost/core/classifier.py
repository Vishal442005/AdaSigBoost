"""Scikit-learn compatible AdaSigBoostClassifier."""

from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d, validate_data

from .base import AdaSigBoostBase
from .losses import BinaryLogLoss, SoftmaxCrossEntropyLoss


class AdaSigBoostClassifier(ClassifierMixin, AdaSigBoostBase):
    """Adaptive Signal-Decomposed Second-Order Gradient Boosting Classifier."""

    def _is_classification(self) -> bool:
        return True

    def _make_loss(self, y: np.ndarray):
        if len(self.classes_) <= 2:
            return BinaryLogLoss()
        return SoftmaxCrossEntropyLoss(n_classes=len(self.classes_))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray | None = None,
    ) -> AdaSigBoostClassifier:
        """Fit AdaSigBoostClassifier on signal data X (and optional tabular covariates Z)."""
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None.")
        y_arr = check_array(y, ensure_2d=False, dtype=None, accept_sparse=False)
        y_arr = column_or_1d(y_arr, warn=True)
        check_classification_targets(y_arr)
        self.classes_, y_encoded = np.unique(y_arr, return_inverse=True)
        if len(self.classes_) == 0:
            raise ValueError("Target y cannot be empty.")

        if len(self.classes_) == 1:
            # Handle degenerate single-class edge case cleanly for sklearn compatibility
            from ..utils.validation import check_signal_and_tabular

            X_3d, Z_arr = check_signal_and_tabular(X, Z)
            self._input_ndim_ = 3 if (isinstance(X, np.ndarray) and X.ndim == 3) else 2
            if self._input_ndim_ == 2:
                validate_data(self, X, accept_sparse=False, dtype=np.float64, reset=True)
            self.n_channels_ = X_3d.shape[1]
            self.signal_length_ = X_3d.shape[2]
            self.n_tabular_features_ = 0 if Z_arr is None else Z_arr.shape[1]
            self.stages_ = []
            self.init_pred_ = np.array([0.0], dtype=np.float64)
            self.n_stages_fitted_ = 0
            self.mode_importances_ = np.zeros(self.n_modes, dtype=np.float64)
            self._single_class_ = True
            return self

        self._single_class_ = False
        self._fit_core(X, y_encoded.astype(np.int64), Z=Z)
        return self

    def decision_function(
        self,
        X: np.ndarray,
        Z: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute raw booster margins F_M(X, Z)."""
        check_is_fitted(self, ["classes_", "init_pred_", "stages_"])
        F, _ = self._raw_predict_and_contributions(X, Z=Z)
        if len(self.classes_) <= 2:
            return F[:, 0]
        return F

    def predict_proba(
        self,
        X: np.ndarray,
        Z: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict class probabilities for signal data X."""
        check_is_fitted(self, ["classes_", "init_pred_", "stages_"])
        if getattr(self, "_single_class_", False):
            from ..utils.validation import check_signal_and_tabular

            X_3d, _ = check_signal_and_tabular(X, Z)
            return np.ones((X_3d.shape[0], 1), dtype=np.float64)
        F, _ = self._raw_predict_and_contributions(X, Z=Z)
        if len(self.classes_) == 2:
            return BinaryLogLoss.predict_proba(F)
        return SoftmaxCrossEntropyLoss.predict_proba(F)

    def predict(
        self,
        X: np.ndarray,
        Z: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict class labels for signal data X."""
        proba = self.predict_proba(X, Z=Z)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Return mean accuracy on (X, y) with optional tabular covariates Z."""
        from sklearn.metrics import accuracy_score

        return float(accuracy_score(y, self.predict(X, Z=Z), sample_weight=sample_weight))
