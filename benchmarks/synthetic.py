"""Controlled multi-component synthetic benchmark generator (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticSignalDataset:
    """Synthetic ground-truth dataset with latent components and frequency bands."""

    X: np.ndarray  # (N, T)
    y: np.ndarray  # (N,)
    components: np.ndarray  # (N, 5, T)
    true_weights: np.ndarray  # (5,)
    snr_db: float
    fs: float
    component_names: tuple[str, ...] = (
        "trend",
        "chirp_threshold",
        "transient_morlet",
        "distractor_1",
        "distractor_2",
    )


def generate_colored_noise(
    rng: np.random.Generator,
    n_samples: int,
    signal_length: int,
    beta: float = 1.0,
) -> np.ndarray:
    """Generate 1/f^beta colored noise normalized to unit variance per sample."""
    white = rng.normal(size=(n_samples, signal_length))
    spec = np.fft.rfft(white, axis=-1)
    freqs = np.fft.rfftfreq(signal_length, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / (freqs[1:] ** (beta / 2.0))
    scale[0] = 0.0
    colored = np.fft.irfft(spec * scale[np.newaxis, :], n=signal_length, axis=-1)
    std = np.std(colored, axis=-1, keepdims=True) + 1e-8
    return colored / std


def generate_synthetic_benchmark(
    n_samples: int = 2000,
    signal_length: int = 1024,
    snr_db: float = 20.0,
    noise_beta: float = 1.0,
    seed: int = 42,
) -> SyntheticSignalDataset:
    """Generate 5-component non-stationary synthetic signal benchmark.

    Components:
      0: Non-stationary polynomial + slow harmonic trend (w* = 1.5)
      1: Frequency-modulated chirp with non-linear threshold (contributes +3.0 iff c_i > 0.7)
      2: Localized Morlet burst at random τ_i ∈ [0.2T, 0.8T] (w* = 2.5)
      3, 4: High-amplitude uncorrelated distractor sinusoids (w* = 0)
    """
    rng = np.random.default_rng(seed)
    fs = float(signal_length)
    t_sec = np.arange(signal_length, dtype=np.float64) / fs  # [0, 1)
    t_norm = t_sec

    a = rng.uniform(-1.5, 1.5, size=n_samples)
    b = rng.uniform(-1.0, 1.0, size=n_samples)
    c = rng.uniform(0.1, 1.3, size=n_samples)
    d = rng.uniform(0.0, 2.0, size=n_samples)
    tau = rng.uniform(0.2, 0.8, size=n_samples)

    # Distractor amplitudes (high energy, zero correlation with y)
    d1_amp = rng.uniform(1.5, 3.5, size=n_samples)
    d2_amp = rng.uniform(1.5, 3.5, size=n_samples)
    d1_phase = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    d2_phase = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)

    # Component 1: Non-stationary trend
    s1 = a[:, None] * (t_norm[None, :] - 0.5) + 0.5 * b[:, None] * np.sin(0.5 * np.pi * t_norm[None, :])

    # Component 2: FM Chirp (sweeping 25 Hz -> 105 Hz)
    f0_chirp, kappa = 25.0, 40.0
    s2 = c[:, None] * np.cos(2.0 * np.pi * (f0_chirp * t_sec[None, :] + kappa * (t_sec[None, :] ** 2)))

    # Component 3: Localized Morlet transient (180 Hz, width ~16 samples -> sigma_t = 8/fs)
    sigma_burst = 8.0 / fs
    f_burst = 180.0
    dt = t_sec[None, :] - tau[:, None]
    s3 = d[:, None] * np.exp(-0.5 * (dt / sigma_burst) ** 2) * np.cos(2.0 * np.pi * f_burst * dt)

    # Components 4 & 5: High-energy uncorrelated distractors (110 Hz and 310 Hz)
    s4 = d1_amp[:, None] * np.cos(2.0 * np.pi * 110.0 * t_sec[None, :] + d1_phase[:, None])
    s5 = d2_amp[:, None] * np.cos(2.0 * np.pi * 310.0 * t_sec[None, :] + d2_phase[:, None])

    components = np.stack([s1, s2, s3, s4, s5], axis=1)
    clean_signal = np.sum(components, axis=1)

    # Scale colored noise to target SNR relative to signal power
    sig_power = np.mean(clean_signal**2, axis=-1, keepdims=True)
    unit_noise = generate_colored_noise(rng, n_samples, signal_length, beta=noise_beta)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    X = clean_signal + np.sqrt(noise_power) * unit_noise

    # Ground-truth regression target
    y = 1.5 * a + 3.0 * (c > 0.7).astype(np.float64) + 2.5 * d

    return SyntheticSignalDataset(
        X=X,
        y=y,
        components=components,
        true_weights=np.array([1.5, 3.0, 2.5, 0.0, 0.0], dtype=np.float64),
        snr_db=snr_db,
        fs=fs,
    )


def compute_spectral_iou(
    s_learned: np.ndarray,
    s_true: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute Spectral Intersection-over-Union between learned mode and ground-truth component.

    s_learned : (N, T)
    s_true    : (N, T)
    """
    p_learn = np.mean(np.abs(np.fft.rfft(s_learned, axis=-1)) ** 2, axis=0)
    p_true = np.mean(np.abs(np.fft.rfft(s_true, axis=-1)) ** 2, axis=0)
    p_learn = p_learn / (np.sum(p_learn) + eps)
    p_true = p_true / (np.sum(p_true) + eps)
    inter = float(np.sum(np.minimum(p_learn, p_true)))
    union = float(np.sum(np.maximum(p_learn, p_true))) + eps
    return inter / union
