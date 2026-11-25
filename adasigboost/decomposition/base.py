"""Data structures and atom-type utilities for composite filterbanks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FilterBankParams:
    """Frozen spectral-spatial filterbank parameters for a single boosting stage."""

    atom_types: tuple[str, ...]
    omega: np.ndarray  # shape (K,) in (0, pi]
    sigma: np.ndarray  # shape (K,) bandwidth > 0
    channel_weights: np.ndarray  # shape (K, C), rows on simplex
    n_channels: int
    signal_length: int


def assign_atom_types(n_modes: int, dictionary_type: str = "composite") -> tuple[str, ...]:
    """Partition K modes across trend, Gabor-Morlet, and rational spectral atoms."""
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}.")
    if dictionary_type == "gabor":
        return ("gabor",) * n_modes
    if dictionary_type == "rational":
        return ("rational",) * n_modes
    if dictionary_type == "composite":
        if n_modes < 3:
            return ("gabor",) * n_modes
        n_gabor = max(1, int(round(0.7 * (n_modes - 1))))
        n_rational = max(0, n_modes - 1 - n_gabor)
        return ("trend",) + ("gabor",) * n_gabor + ("rational",) * n_rational
    raise ValueError(f"Unknown dictionary_type: {dictionary_type!r}")
