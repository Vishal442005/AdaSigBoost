"""Twice-differentiable convex loss functions for second-order boosting."""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_l = np.exp(np.clip(shifted, -30.0, 30.0))
    return exp_l / np.sum(exp_l, axis=1, keepdims=True)


class SquaredLoss:
    """L2 regression loss: 0.5 * (y - F)^2."""

    n_outputs = 1

    def init_prediction(self, y: np.ndarray) -> np.ndarray:
        return np.array([float(np.mean(y))], dtype=np.float64)

    def loss(self, y: np.ndarray, F: np.ndarray) -> float:
        diff = F[:, 0] - y
        return float(0.5 * np.mean(diff * diff))

    def grad_hess(self, y: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        g = F[:, 0] - y
        h = np.ones_like(g)
        return g[:, np.newaxis], h[:, np.newaxis]


class HuberLoss:
    """Pseudo-Huber regression loss with smooth second derivative."""

    n_outputs = 1

    def __init__(self, delta: float = 1.35) -> None:
        self.delta = float(delta)

    def init_prediction(self, y: np.ndarray) -> np.ndarray:
        return np.array([float(np.median(y))], dtype=np.float64)

    def loss(self, y: np.ndarray, F: np.ndarray) -> float:
        r = F[:, 0] - y
        d = self.delta
        return float(np.mean((d**2) * (np.sqrt(1.0 + (r / d) ** 2) - 1.0)))

    def grad_hess(self, y: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = F[:, 0] - y
        d = self.delta
        denom = np.sqrt(1.0 + (r / d) ** 2)
        g = r / denom
        h = np.maximum(1.0 / (denom**3), 1e-4)
        return g[:, np.newaxis], h[:, np.newaxis]


class BinaryLogLoss:
    """Binary cross-entropy loss on raw log-odds F."""

    n_outputs = 1

    def init_prediction(self, y: np.ndarray) -> np.ndarray:
        p = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
        return np.array([np.log(p / (1.0 - p))], dtype=np.float64)

    def loss(self, y: np.ndarray, F: np.ndarray) -> float:
        f = np.clip(F[:, 0], -30.0, 30.0)
        return float(np.mean(np.logaddexp(0.0, f) - y * f))

    def grad_hess(self, y: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = _sigmoid(F[:, 0])
        g = p - y
        h = np.maximum(p * (1.0 - p), 1e-6)
        return g[:, np.newaxis], h[:, np.newaxis]

    @staticmethod
    def predict_proba(F: np.ndarray) -> np.ndarray:
        p1 = _sigmoid(F[:, 0])
        return np.column_stack([1.0 - p1, p1])


class SoftmaxCrossEntropyLoss:
    """Multi-class cross-entropy loss with diagonal Hessian approximation."""

    def __init__(self, n_classes: int) -> None:
        self.n_outputs = int(n_classes)

    def init_prediction(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y.astype(np.int64), minlength=self.n_outputs).astype(np.float64)
        priors = np.clip(counts / max(float(len(y)), 1.0), 1e-4, 1.0)
        logits = np.log(priors)
        return logits - np.mean(logits)

    def loss(self, y: np.ndarray, F: np.ndarray) -> float:
        probs = np.clip(_softmax(F), 1e-12, 1.0)
        idx = np.arange(len(y))
        return float(-np.mean(np.log(probs[idx, y.astype(np.int64)])))

    def grad_hess(self, y: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probs = _softmax(F)
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(y)), y.astype(np.int64)] = 1.0
        g = probs - onehot
        h = np.maximum(probs * (1.0 - probs), 1e-6)
        return g, h

    @staticmethod
    def predict_proba(F: np.ndarray) -> np.ndarray:
        return _softmax(F)
