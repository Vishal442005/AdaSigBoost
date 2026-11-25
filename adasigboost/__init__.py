"""AdaSigBoost: Task-Driven Adaptive Signal Decomposition with Joint Second-Order Gradient Boosting."""

__version__ = "0.1.0"

from typing import Any

_adasigboost_cpp: Any
try:
    from . import _adasigboost_cpp as _adasigboost_cpp  # type: ignore[no-redef]
except ImportError:  # pragma: no cover
    _adasigboost_cpp = None

from .core import AdaSigBoostClassifier, AdaSigBoostRegressor  # noqa: E402

__all__ = [
    "__version__",
    "_adasigboost_cpp",
    "AdaSigBoostClassifier",
    "AdaSigBoostRegressor",
]
