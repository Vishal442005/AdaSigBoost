"""Adaptive multi-physics signal decomposition layers."""

from .base import FilterBankParams, assign_atom_types
from .filterbank import (
    apply_filterbank_np,
    apply_filterbank_torch,
    compute_residual_spectral_peaks,
    fit_stage_filterbank,
)
from .regularizers import decomposition_penalty_torch

__all__ = [
    "FilterBankParams",
    "assign_atom_types",
    "apply_filterbank_np",
    "apply_filterbank_torch",
    "compute_residual_spectral_peaks",
    "fit_stage_filterbank",
    "decomposition_penalty_torch",
]
