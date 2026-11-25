"""Differentiable summary statistics Φ̃(s) for Block 1 filterbank optimization."""

from __future__ import annotations

import torch


def extract_differentiable_stats(
    s_mix: torch.Tensor,
    mode: str = "full",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute fast smooth differentiable descriptors Φ̃(s_mix) of shape (N, K, D_stats)."""
    N, K, T = s_mix.shape
    energy = torch.mean(s_mix * s_mix, dim=-1)
    log_energy = torch.log(energy + eps)

    if mode == "energy_only":
        return log_energy.unsqueeze(-1)

    # Partition T into B <= 16 temporal blocks for O(T) block-RMS envelope & burst detection
    n_blocks = min(16, max(1, T // 4)) if T >= 4 else 1
    blk_len = T // n_blocks
    if n_blocks >= 2 and blk_len >= 1:
        trimmed = s_mix[..., : n_blocks * blk_len].reshape(N, K, n_blocks, blk_len)
        blk_e = torch.mean(trimmed * trimmed, dim=-1)  # (N, K, B)
        env = torch.sqrt(blk_e + eps)
    else:
        blk_e = energy.unsqueeze(-1)
        env = torch.sqrt(blk_e + eps)

    env_mean = torch.mean(env, dim=-1)
    env_std = torch.sqrt(torch.var(env, dim=-1, unbiased=False) + eps)

    # Smooth-max block envelope via LogSumExp over <= 16 blocks
    tau = 15.0
    env_scale = torch.clamp(env_mean, min=1e-4)
    scaled_env = env / env_scale.unsqueeze(-1)
    smooth_max = (
        env_scale
        * (
            torch.logsumexp(tau * torch.clamp(scaled_env, max=20.0), dim=-1)
            - torch.log(torch.tensor(float(env.shape[-1]), dtype=s_mix.dtype, device=s_mix.device))
        )
        / tau
    )

    burst_ratio = torch.max(blk_e, dim=-1).values / (torch.mean(blk_e, dim=-1) + eps)

    if T > 1:
        diff = s_mix[..., 1:] - s_mix[..., :-1]
        flux = torch.log(torch.mean(diff * diff, dim=-1) + eps)
        t_cent = torch.linspace(-0.5, 0.5, T, dtype=s_mix.dtype, device=s_mix.device)
        slope = torch.sum(s_mix * t_cent, dim=-1) / (torch.sum(t_cent * t_cent) + eps)
    else:
        flux = log_energy
        slope = torch.zeros_like(log_energy)

    signed_mean = torch.mean(s_mix, dim=-1)
    return torch.stack(
        [log_energy, signed_mean, slope, env_mean, env_std, smooth_max, burst_ratio, flux], dim=-1
    )
