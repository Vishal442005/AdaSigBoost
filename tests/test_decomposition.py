"""Unit tests for composite filterbanks, differentiable stats, and peak init."""

from __future__ import annotations

import numpy as np
import torch

from adasigboost.decomposition import (
    FilterBankParams,
    apply_filterbank_np,
    apply_filterbank_torch,
    assign_atom_types,
    compute_residual_spectral_peaks,
    decomposition_penalty_torch,
)
from adasigboost.descriptors import extract_descriptors, extract_differentiable_stats


def test_filterbank_norm_and_gradient_flow() -> None:
    rng = np.random.default_rng(0)
    N, C, T, K = 16, 2, 128, 4
    X = rng.normal(size=(N, C, T))
    X_fft_np = np.fft.rfft(X, axis=-1)
    X_fft_t = torch.tensor(X_fft_np, dtype=torch.complex128)

    atom_types = assign_atom_types(K, "composite")
    omega = torch.tensor([0.1, 0.4, 1.2, 2.1], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([0.1, 0.2, 0.2, 0.15], dtype=torch.float64, requires_grad=True)
    ch_logits = torch.zeros((K, C), dtype=torch.float64, requires_grad=True)
    ch_w = torch.softmax(ch_logits, dim=-1)

    s_ch, s_mix = apply_filterbank_torch(X_fft_t, atom_types, omega, sigma, ch_w, T)
    assert s_ch.shape == (N, K, C, T)
    assert s_mix.shape == (N, K, T)

    stats = extract_differentiable_stats(s_mix, mode="full")
    assert stats.shape == (N, K, 8)
    reg = decomposition_penalty_torch(s_mix, alpha1=0.05, alpha2=0.01)
    loss = torch.sum(stats) + reg
    loss.backward()

    assert omega.grad is not None and torch.all(torch.isfinite(omega.grad))
    assert sigma.grad is not None and torch.all(torch.isfinite(sigma.grad))
    assert ch_logits.grad is not None and torch.all(torch.isfinite(ch_logits.grad))

    # Parity check with NumPy path
    params = FilterBankParams(
        atom_types=atom_types,
        omega=omega.detach().numpy(),
        sigma=sigma.detach().numpy(),
        channel_weights=ch_w.detach().numpy(),
        n_channels=C,
        signal_length=T,
    )
    s_ch_np, s_mix_np = apply_filterbank_np(X_fft_np, params)
    np.testing.assert_allclose(s_ch_np, s_ch.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(s_mix_np, s_mix.detach().numpy(), atol=1e-10)

    desc = extract_descriptors(s_mix_np, s_ch=s_ch_np)
    assert desc.shape == (N, K, 19)
    assert np.all(np.isfinite(desc))


def test_cross_spectral_peak_detection() -> None:
    rng = np.random.default_rng(1)
    N, T = 64, 256
    t = np.arange(T)
    w_true = 0.6 * np.pi
    amp = rng.uniform(0.5, 2.0, size=N)
    X = amp[:, None] * np.sin(w_true * t)[None, :] + 0.05 * rng.normal(size=(N, T))
    X_fft = np.fft.rfft(X[:, None, :], axis=-1)
    X_power = np.mean(X_fft.real**2 + X_fft.imag**2, axis=1)
    g = -(amp - np.mean(amp))

    peaks = compute_residual_spectral_peaks(X_power, g, n_peaks=1, omega_min=0.05, omega_max=0.95 * np.pi)
    assert abs(peaks[0] - w_true) < 0.08
