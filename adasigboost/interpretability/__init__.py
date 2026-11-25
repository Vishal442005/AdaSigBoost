"""Interpretability metrics, physical spectral alignment, and publication figures."""

from .mode_attribution import (
    compute_mode_deletion_auc,
    extract_learned_frequencies,
    spectral_frequency_alignment_error,
    summarize_stage_modes,
)
from .plots import (
    plot_critical_difference_approx,
    plot_mode_decomposition,
    plot_snr_benchmark,
)

__all__ = [
    "compute_mode_deletion_auc",
    "extract_learned_frequencies",
    "spectral_frequency_alignment_error",
    "summarize_stage_modes",
    "plot_mode_decomposition",
    "plot_snr_benchmark",
    "plot_critical_difference_approx",
]
