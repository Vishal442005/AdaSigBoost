"""Composite multi-physics spectral filterbanks and Block 1 OOF solver."""

from __future__ import annotations

import numpy as np
import torch

from ..descriptors.differentiable_stats import extract_differentiable_stats
from .base import FilterBankParams, assign_atom_types
from .regularizers import decomposition_penalty_torch


def _dft_weights_np(n_freq: int, T: int) -> np.ndarray:
    w = np.full(n_freq, 2.0, dtype=np.float64)
    w[0] = 1.0
    if T % 2 == 0 and n_freq > 1:
        w[-1] = 1.0
    return w


def _dft_weights_torch(n_freq: int, T: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    w = torch.full((n_freq,), 2.0, dtype=dtype, device=device)
    w[0] = 1.0
    if T % 2 == 0 and n_freq > 1:
        w[-1] = 1.0
    return w


def build_filter_masks_np(
    atom_types: tuple[str, ...],
    omega: np.ndarray,
    sigma: np.ndarray,
    T: int,
    eps_norm: float = 1e-8,
) -> np.ndarray:
    """Construct unit-L2-normalized frequency responses H_k(ω) of shape (K, F)."""
    n_freq = T // 2 + 1
    freqs = np.linspace(0.0, np.pi, n_freq, dtype=np.float64)
    K = len(atom_types)
    H = np.zeros((K, n_freq), dtype=np.float64)
    w_dft = _dft_weights_np(n_freq, T)

    for k, atype in enumerate(atom_types):
        om = float(omega[k])
        sg = max(float(sigma[k]), 1e-3)
        if atype == "trend":
            raw = 1.0 / np.sqrt(1.0 + (freqs / max(om, 1e-3)) ** 4)
        elif atype == "rational":
            raw = 1.0 / (1.0 + ((freqs - om) / sg) ** 2) + 1.0 / (
                1.0 + ((freqs + om) / sg) ** 2
            )
        else:
            raw = np.exp(-0.5 * ((freqs - om) / sg) ** 2) + np.exp(
                -0.5 * ((freqs + om) / sg) ** 2
            )
        norm = np.sqrt(np.sum(w_dft * (raw * raw)) / max(float(T), 1.0) + eps_norm)
        H[k] = raw / norm
    return H


def build_filter_masks_torch(
    atom_types: tuple[str, ...],
    omega: torch.Tensor,
    sigma: torch.Tensor,
    T: int,
    eps_norm: float = 1e-8,
) -> torch.Tensor:
    """Differentiable unit-L2-normalized frequency masks H_k(ω) of shape (K, F)."""
    n_freq = T // 2 + 1
    freqs = torch.linspace(0.0, np.pi, n_freq, dtype=omega.dtype, device=omega.device)
    w_dft = _dft_weights_torch(n_freq, T, omega.dtype, omega.device)
    masks = []
    for k, atype in enumerate(atom_types):
        om = omega[k]
        sg = torch.clamp(sigma[k], min=1e-3)
        if atype == "trend":
            raw = 1.0 / torch.sqrt(1.0 + (freqs / torch.clamp(om, min=1e-3)) ** 4)
        elif atype == "rational":
            raw = 1.0 / (1.0 + ((freqs - om) / sg) ** 2) + 1.0 / (
                1.0 + ((freqs + om) / sg) ** 2
            )
        else:
            raw = torch.exp(-0.5 * ((freqs - om) / sg) ** 2) + torch.exp(
                -0.5 * ((freqs + om) / sg) ** 2
            )
        norm = torch.sqrt(torch.sum(w_dft * (raw * raw)) / max(float(T), 1.0) + eps_norm)
        masks.append(raw / norm)
    return torch.stack(masks, dim=0)


def apply_filterbank_np(
    X_fft: np.ndarray,
    params: FilterBankParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply frozen filterbank in pure NumPy. Returns (s_ch, s_mix)."""
    T = params.signal_length
    H = build_filter_masks_np(params.atom_types, params.omega, params.sigma, T)
    S_fft = X_fft[:, np.newaxis, :, :] * H[np.newaxis, :, np.newaxis, :]
    s_ch = np.fft.irfft(S_fft, n=T, axis=-1)
    s_mix = np.einsum("kc,nkct->nkt", params.channel_weights, s_ch)
    return s_ch, s_mix


def apply_filterbank_torch(
    X_fft: torch.Tensor,
    atom_types: tuple[str, ...],
    omega: torch.Tensor,
    sigma: torch.Tensor,
    channel_weights: torch.Tensor,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable filterbank application in PyTorch."""
    H = build_filter_masks_torch(atom_types, omega, sigma, T)
    S_fft = X_fft.unsqueeze(1) * H.unsqueeze(0).unsqueeze(2)
    s_ch = torch.fft.irfft(S_fft, n=T, dim=-1)
    s_mix = torch.einsum("kc,nkct->nkt", channel_weights, s_ch)
    return s_ch, s_mix


def compute_residual_spectral_peaks(
    X_power: np.ndarray,
    g: np.ndarray,
    n_peaks: int,
    omega_min: float,
    omega_max: float,
) -> np.ndarray:
    """Initialize frequencies at top peaks of h/g-weighted cross-spectral residual density."""
    if n_peaks <= 0:
        return np.empty(0, dtype=np.float64)
    n_freq = X_power.shape[1]
    freqs = np.linspace(0.0, np.pi, n_freq, dtype=np.float64)
    g_2d = g[:, np.newaxis] if g.ndim == 1 else g

    g_centered = g_2d - np.mean(g_2d, axis=0, keepdims=True)
    density = np.sum(np.abs(g_centered.T @ X_power), axis=0)

    valid = (freqs >= omega_min) & (freqs <= omega_max)
    if not np.any(valid):
        return np.geomspace(max(omega_min, 1e-3), max(omega_max, 2e-3), n_peaks)

    masked = np.where(valid, density, -1.0)
    min_sep = max((omega_max - omega_min) / (2.2 * max(n_peaks, 1)), np.pi / max(n_freq, 2))
    chosen: list[float] = []
    for idx in np.argsort(masked)[::-1]:
        if masked[idx] <= 0.0:
            break
        f_cand = float(freqs[idx])
        if all(abs(f_cand - f_prev) >= min_sep for f_prev in chosen):
            chosen.append(f_cand)
            if len(chosen) == n_peaks:
                break

    if len(chosen) < n_peaks:
        grid = np.geomspace(max(omega_min, 1e-3), max(omega_max, 2e-3), n_peaks + 2)[1:-1]
        for f_grid in grid:
            if len(chosen) == n_peaks:
                break
            if all(abs(float(f_grid) - f_prev) >= 0.5 * min_sep for f_prev in chosen):
                chosen.append(float(f_grid))
        while len(chosen) < n_peaks:
            chosen.append(float(np.mean([omega_min, omega_max])))

    return np.sort(np.asarray(chosen, dtype=np.float64))


def _weighted_pearson_sq(
    stats: torch.Tensor,
    r: torch.Tensor,
    h: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute sum_{k,c,d} Corr_h(stats[:, k, d], r[:, c])^2."""
    h_norm = h / (torch.sum(h, dim=0, keepdim=True) + eps)
    r_mean = torch.sum(h_norm * r, dim=0, keepdim=True)
    r_cent = r - r_mean
    r_var = torch.sum(h_norm * (r_cent * r_cent), dim=0) + eps

    s_mean = torch.einsum("nc,nkd->ckd", h_norm, stats)
    s_cent = stats.unsqueeze(1) - s_mean.unsqueeze(0)
    s_var = torch.einsum("nc,nckd->ckd", h_norm, s_cent * s_cent) + eps

    cov = torch.einsum("nc,nckd->ckd", h_norm * r_cent, s_cent)
    corr_sq = (cov * cov) / (r_var[:, None, None] * s_var + eps)
    return torch.sum(corr_sq)


def _optimize_single_filterbank(
    X_fft_train: torch.Tensor,
    r_train: torch.Tensor,
    h_train: torch.Tensor,
    atom_types: tuple[str, ...],
    omega_init: np.ndarray,
    sigma_init: np.ndarray,
    ch_init: np.ndarray,
    omega_min: float,
    omega_max: float,
    sigma_min: float,
    sigma_max: float,
    n_channels: int,
    signal_length: int,
    alpha1: float,
    alpha2: float,
    n_steps: int,
    lr: float,
    block1_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimize filterbank parameters on a single training subset."""
    dtype = X_fft_train.real.dtype
    device = X_fft_train.device
    n_modes = len(atom_types)

    def _to_logit(val: np.ndarray, lo: float, hi: float) -> torch.Tensor:
        p = np.clip((val - lo) / max(hi - lo, 1e-6), 1e-3, 1.0 - 1e-3)
        return torch.tensor(np.log(p / (1.0 - p)), dtype=dtype, device=device)

    omega_raw = _to_logit(omega_init, omega_min, omega_max).requires_grad_(True)
    sigma_raw = _to_logit(sigma_init, sigma_min, sigma_max).requires_grad_(True)
    params_to_opt = [omega_raw, sigma_raw]

    if n_channels > 1:
        ch_raw = torch.tensor(
            np.log(np.clip(ch_init, 1e-4, 1.0)),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        params_to_opt.append(ch_raw)
    else:
        ch_raw = None

    # Lightweight manual Adam loop (avoids torch._dynamo import overhead in torch.optim.Adam)
    m_states = [torch.zeros_like(p) for p in params_to_opt]
    v_states = [torch.zeros_like(p) for p in params_to_opt]
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    for step_idx in range(1, n_steps + 1):
        for p in params_to_opt:
            if p.grad is not None:
                p.grad.zero_()
        omega = omega_min + (omega_max - omega_min) * torch.sigmoid(omega_raw)
        sigma = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(sigma_raw)
        if ch_raw is not None:
            ch_w = torch.softmax(ch_raw, dim=-1)
        else:
            ch_w = torch.ones((n_modes, 1), dtype=dtype, device=device)

        _, s_mix = apply_filterbank_torch(
            X_fft_train, atom_types, omega, sigma, ch_w, signal_length
        )
        stats = extract_differentiable_stats(s_mix, mode=block1_mode)
        align = _weighted_pearson_sq(stats, r_train, h_train)
        reg = decomposition_penalty_torch(s_mix, alpha1=alpha1, alpha2=alpha2)
        loss = -align + reg
        loss.backward()

        bc1 = 1.0 - beta1**step_idx
        bc2 = 1.0 - beta2**step_idx
        with torch.no_grad():
            for idx_p, p in enumerate(params_to_opt):
                if p.grad is None:
                    continue
                g_p = p.grad
                m_states[idx_p].mul_(beta1).add_(g_p, alpha=1.0 - beta1)
                v_states[idx_p].mul_(beta2).addcmul_(g_p, g_p, value=1.0 - beta2)
                m_hat = m_states[idx_p] / bc1
                v_hat = v_states[idx_p] / bc2
                p.sub_(lr * m_hat / (torch.sqrt(v_hat) + adam_eps))

    with torch.no_grad():
        omega_out = (
            (omega_min + (omega_max - omega_min) * torch.sigmoid(omega_raw))
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        sigma_out = (
            (sigma_min + (sigma_max - sigma_min) * torch.sigmoid(sigma_raw))
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        if ch_raw is not None:
            ch_out = torch.softmax(ch_raw, dim=-1).cpu().numpy().astype(np.float64)
        else:
            ch_out = np.ones((n_modes, 1), dtype=np.float64)
    return omega_out, sigma_out, ch_out


def fit_stage_filterbank(
    X_fft_np: np.ndarray,
    X_fft_torch: torch.Tensor,
    X_power_np: np.ndarray,
    g_np: np.ndarray,
    h_np: np.ndarray,
    folds: np.ndarray,
    n_modes: int,
    n_channels: int,
    signal_length: int,
    dictionary_type: str = "composite",
    prev_params: FilterBankParams | None = None,
    active_mask: np.ndarray | None = None,
    alpha1: float = 0.05,
    alpha2: float = 0.01,
    n_steps: int = 16,
    lr: float = 0.08,
    use_oof: bool = True,
    freeze_init: bool = False,
    block1_mode: str = "full",
) -> tuple[FilterBankParams, np.ndarray, np.ndarray]:
    """Solve Block 1 via honest Q-fold OOF cross-fitting + residual spectral peak initialization.

    Returns
    -------
    params : FilterBankParams
        Consensus filterbank parameters for test-time inference.
    s_ch_train : ndarray of shape (N, K, C, T)
        Honest out-of-fold per-channel decomposed sub-signals for training.
    s_mix_train : ndarray of shape (N, K, T)
        Honest out-of-fold channel-mixed decomposed sub-signals for training.
    """
    N = X_fft_np.shape[0]
    atom_types = assign_atom_types(n_modes, dictionary_type)
    omega_min = float(np.pi / max(signal_length, 4))
    omega_max = float(0.95 * np.pi)
    sigma_min = float(max(0.015, 0.5 * omega_min))
    sigma_max = float(0.65 * np.pi)

    omega_init = np.zeros(n_modes, dtype=np.float64)
    sigma_init = np.full(n_modes, 0.12 * np.pi, dtype=np.float64)

    n_non_trend = sum(1 for a in atom_types if a != "trend")
    if freeze_init:
        peaks = np.geomspace(max(omega_min, 1e-3), omega_max, max(n_non_trend, 1))
    else:
        peaks = compute_residual_spectral_peaks(
            X_power_np, g_np, max(n_non_trend, 1), omega_min, omega_max
        )

    peak_idx = 0
    for k, atype in enumerate(atom_types):
        if (
            not freeze_init
            and prev_params is not None
            and active_mask is not None
            and active_mask[k]
        ):
            omega_init[k] = float(np.clip(prev_params.omega[k], omega_min * 1.05, omega_max * 0.98))
            sigma_init[k] = float(np.clip(prev_params.sigma[k], sigma_min * 1.05, sigma_max * 0.95))
        elif atype == "trend":
            omega_init[k] = float(np.clip(0.05 * np.pi, omega_min * 1.1, 0.2 * np.pi))
            sigma_init[k] = float(np.clip(0.05 * np.pi, sigma_min, sigma_max))
        else:
            omega_init[k] = float(
                np.clip(peaks[min(peak_idx, len(peaks) - 1)], omega_min * 1.05, omega_max * 0.98)
            )
            sigma_init[k] = float(
                np.clip(0.08 * np.pi if atype == "rational" else 0.14 * np.pi, sigma_min, sigma_max)
            )
            peak_idx += 1

    if prev_params is not None and active_mask is not None:
        ch_init = prev_params.channel_weights.copy()
    else:
        ch_init = np.full((n_modes, n_channels), 1.0 / n_channels, dtype=np.float64)

    if freeze_init or n_steps <= 0 or signal_length <= 2:
        params = FilterBankParams(
            atom_types=atom_types,
            omega=omega_init,
            sigma=sigma_init,
            channel_weights=ch_init,
            n_channels=n_channels,
            signal_length=signal_length,
        )
        s_ch, s_mix = apply_filterbank_np(X_fft_np, params)
        return params, s_ch, s_mix

    g_2d = g_np[:, np.newaxis] if g_np.ndim == 1 else g_np
    h_2d = h_np[:, np.newaxis] if h_np.ndim == 1 else h_np
    r_np = -g_2d / (h_2d + 1e-6)
    r_std = np.std(r_np, axis=0, keepdims=True) + 1e-6
    r_np = np.clip(r_np, -6.0 * r_std, 6.0 * r_std)

    dtype = X_fft_torch.real.dtype
    device = X_fft_torch.device
    r_t = torch.tensor(r_np, dtype=dtype, device=device)
    h_t = torch.tensor(np.maximum(h_2d, 1e-6), dtype=dtype, device=device)

    unique_folds = np.unique(folds)
    if not use_oof or len(unique_folds) < 2:
        om_f, sg_f, ch_f = _optimize_single_filterbank(
            X_fft_torch,
            r_t,
            h_t,
            atom_types,
            omega_init,
            sigma_init,
            ch_init,
            omega_min,
            omega_max,
            sigma_min,
            sigma_max,
            n_channels,
            signal_length,
            alpha1,
            alpha2,
            n_steps,
            lr,
            block1_mode,
        )
        params = FilterBankParams(
            atom_types=atom_types,
            omega=om_f,
            sigma=sg_f,
            channel_weights=ch_f,
            n_channels=n_channels,
            signal_length=signal_length,
        )
        s_ch, s_mix = apply_filterbank_np(X_fft_np, params)
        return params, s_ch, s_mix

    # Honest Q-fold OOF cross-fitting: fold q is decomposed using filterbank fitted on != q
    s_ch_oof = np.zeros((N, n_modes, n_channels, signal_length), dtype=np.float64)
    s_mix_oof = np.zeros((N, n_modes, signal_length), dtype=np.float64)
    omegas_fold = []
    sigmas_fold = []
    chs_fold = []

    for q in unique_folds:
        val_mask = folds == q
        tr_mask = ~val_mask
        tr_indices = np.flatnonzero(tr_mask)
        # Subsample up to 160 samples per fold for fast O(1) filter parameter optimization
        if len(tr_indices) > 160:
            step = len(tr_indices) // 160
            tr_indices = tr_indices[::step][:160]
        tr_idx = torch.tensor(tr_indices, dtype=torch.long, device=device)
        om_q, sg_q, ch_q = _optimize_single_filterbank(
            X_fft_torch[tr_idx],
            r_t[tr_idx],
            h_t[tr_idx],
            atom_types,
            omega_init,
            sigma_init,
            ch_init,
            omega_min,
            omega_max,
            sigma_min,
            sigma_max,
            n_channels,
            signal_length,
            alpha1,
            alpha2,
            n_steps,
            lr,
            block1_mode,
        )
        omegas_fold.append(om_q)
        sigmas_fold.append(sg_q)
        chs_fold.append(ch_q)

        fold_params = FilterBankParams(
            atom_types=atom_types,
            omega=om_q,
            sigma=sg_q,
            channel_weights=ch_q,
            n_channels=n_channels,
            signal_length=signal_length,
        )
        s_ch_q, s_mix_q = apply_filterbank_np(X_fft_np[val_mask], fold_params)
        s_ch_oof[val_mask] = s_ch_q
        s_mix_oof[val_mask] = s_mix_q

    # Consensus parameters for inference
    ch_mean = np.mean(np.stack(chs_fold, axis=0), axis=0)
    ch_mean /= np.sum(ch_mean, axis=-1, keepdims=True) + 1e-12
    consensus_params = FilterBankParams(
        atom_types=atom_types,
        omega=np.mean(np.stack(omegas_fold, axis=0), axis=0),
        sigma=np.mean(np.stack(sigmas_fold, axis=0), axis=0),
        channel_weights=ch_mean,
        n_channels=n_channels,
        signal_length=signal_length,
    )
    return consensus_params, s_ch_oof, s_mix_oof
