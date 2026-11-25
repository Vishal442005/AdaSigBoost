"""Deterministic seeding across NumPy and PyTorch."""

from __future__ import annotations

import numpy as np
import torch


def set_deterministic_seed(seed: int | None) -> np.random.Generator:
    """Return a deterministic NumPy Generator and seed PyTorch if seed is provided."""
    if seed is not None:
        torch.manual_seed(int(seed))
    return np.random.default_rng(seed)
