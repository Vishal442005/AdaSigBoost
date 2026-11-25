"""Vectorized physical time-frequency and cross-channel descriptors Φ(s)."""

from __future__ import annotations

import numpy as np

DESCRIPTOR_NAMES: tuple[str, ...] = (
    "log_energy",
    "energy_ratio",
    "signed_mean",
    "trend_slope",
    "env_mean",
    "env_std",
    "env_p90",
    "env_max",
    "env_kurtosis",
    "env_skewness",
    "crest_factor",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "dominant_freq",
    "dominant_phase_cos",
    "dominant_phase_sin",
    "local_energy_concentration",
    "cross_channel_coherence",
)


def _analytic_envelope_np(s: np.ndarray) -> np.ndarray:
    """Compute Hilbert amplitude envelope along the last axis of real array s (..., T)."""
    T = s.shape[-1]
    if T <= 1:
        return np.abs(s)
    Xf = np.fft.fft(s, axis=-1)
    h = np.zeros(T, dtype=np.float64)
    if T % 2 == 0:
        h[0] = 1.0
        h[T // 2] = 1.0
        h[1 : T // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (T + 1) // 2] = 2.0
    analytic = np.fft.ifft(Xf * h, axis=-1)
    return np.abs(analytic)


def extract_descriptors(
    s_mix: np.ndarray,
    s_ch: np.ndarray | None = None,
    raw_energy: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Extract d_Φ = 16 invariant descriptors per (sample, mode).

    Parameters
    ----------
    s_mix : ndarray of shape (N, K, T)
        Channel-mixed decomposed sub-signals.
    s_ch : ndarray of shape (N, K, C, T), optional
        Per-channel sub-signals used to compute cross-channel coherence.
    raw_energy : ndarray of shape (N,), optional
        Mean squared energy of the raw input signal per sample.

    Returns
    -------
    features : ndarray of shape (N, K, 16)
    """
    N, K, T = s_mix.shape
    out = np.zeros((N, K, len(DESCRIPTOR_NAMES)), dtype=np.float64)

    energy = np.mean(s_mix * s_mix, axis=-1)  # (N, K)
    rms = np.sqrt(energy + eps)

    # 0. Log energy
    out[:, :, 0] = np.log(energy + eps)

    # 1. Energy ratio vs raw input
    if raw_energy is not None:
        out[:, :, 1] = energy / (raw_energy[:, np.newaxis] + eps)
    else:
        tot = np.sum(energy, axis=1, keepdims=True) + eps
        out[:, :, 1] = energy / tot

    # 2-3. Signed mean and linear trend slope (captures polarity of DC/drift modes)
    out[:, :, 2] = np.mean(s_mix, axis=-1)
    if T > 1:
        t_centered = np.linspace(-0.5, 0.5, T, dtype=np.float64)
        denom_t = float(np.sum(t_centered * t_centered)) + eps
        out[:, :, 3] = np.sum(s_mix * t_centered[np.newaxis, np.newaxis, :], axis=-1) / denom_t
    else:
        out[:, :, 3] = 0.0

    # 4-9. Hilbert envelope statistics
    env = _analytic_envelope_np(s_mix)  # (N, K, T)
    env_mean = np.mean(env, axis=-1)
    env_centered = env - env_mean[:, :, np.newaxis]
    env_var = np.mean(env_centered * env_centered, axis=-1)
    env_std = np.sqrt(env_var + eps)
    z_env = env_centered / env_std[:, :, np.newaxis]

    out[:, :, 4] = env_mean
    out[:, :, 5] = env_std
    out[:, :, 6] = np.percentile(env, 90.0, axis=-1)
    out[:, :, 7] = np.max(env, axis=-1)
    out[:, :, 8] = np.mean(z_env**4, axis=-1) - 3.0
    out[:, :, 9] = np.mean(z_env**3, axis=-1)

    # 10. Crest factor
    out[:, :, 10] = np.max(np.abs(s_mix), axis=-1) / rms

    # 11. Zero-crossing rate
    if T > 1:
        signs = np.signbit(s_mix)
        zcr = np.mean(signs[:, :, 1:] != signs[:, :, :-1], axis=-1)
        out[:, :, 11] = zcr
    else:
        out[:, :, 11] = 0.0

    # 12-16. Spectral centroid, bandwidth, dominant frequency & phase
    spec = np.fft.rfft(s_mix, axis=-1)  # (N, K, F)
    mag_sq = spec.real**2 + spec.imag**2
    n_freq = spec.shape[-1]
    freqs = np.linspace(0.0, 1.0, n_freq, dtype=np.float64)
    mag_sum = np.sum(mag_sq, axis=-1) + eps
    centroid = np.sum(mag_sq * freqs[np.newaxis, np.newaxis, :], axis=-1) / mag_sum
    bw = np.sqrt(
        np.sum(
            mag_sq * (freqs[np.newaxis, np.newaxis, :] - centroid[:, :, np.newaxis]) ** 2,
            axis=-1,
        )
        / mag_sum
        + eps
    )
    dom_idx = np.argmax(mag_sq, axis=-1)
    dom_freq = freqs[dom_idx]
    dom_spec = np.take_along_axis(spec, dom_idx[:, :, np.newaxis], axis=-1)[..., 0]
    dom_abs = np.abs(dom_spec) + eps
    dom_cos = dom_spec.real / dom_abs
    dom_sin = dom_spec.imag / dom_abs

    out[:, :, 12] = centroid
    out[:, :, 13] = bw
    out[:, :, 14] = dom_freq
    out[:, :, 15] = dom_cos
    out[:, :, 16] = dom_sin

    # 17. Temporal energy concentration across 4 segments
    if T >= 4:
        n_win = 4
        win_len = T // n_win
        trimmed = s_mix[:, :, : n_win * win_len].reshape(N, K, n_win, win_len)
        win_e = np.mean(trimmed * trimmed, axis=-1)  # (N, K, 4)
        out[:, :, 17] = np.max(win_e, axis=-1) / (np.mean(win_e, axis=-1) + eps)
    else:
        out[:, :, 17] = 1.0

    # 18. Cross-channel coherence
    if s_ch is not None and s_ch.shape[2] > 1:
        ch_energy = np.mean(np.mean(s_ch * s_ch, axis=-1), axis=-1)  # (N, K)
        out[:, :, 18] = energy / (ch_energy + eps)
    else:
        out[:, :, 18] = 1.0

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
