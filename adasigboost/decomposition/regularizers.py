"""Decomposition regularization penalties: mutual incoherence and temporal smoothness."""

from __future__ import annotations

import torch


def decomposition_penalty_torch(
    s_mix: torch.Tensor,
    alpha1: float = 0.05,
    alpha2: float = 0.01,
) -> torch.Tensor:
    """Compute Ω_decomp(Θ_m) = α1 * sum_{j<k} E[<s_j, s_k>^2] + α2 * sum_k E[||∇_t s_k||^2].

    Parameters
    ----------
    s_mix : Tensor of shape (N, K, T)
    """
    _, K, T = s_mix.shape
    penalty = torch.zeros((), dtype=s_mix.dtype, device=s_mix.device)

    if alpha1 > 0.0 and K > 1:
        # Normalize inner product by T: <s_j, s_k> = (1/T) sum_t s_{i,j,t} s_{i,k,t}
        gram = torch.einsum("nkt,nlt->nkl", s_mix, s_mix) / float(T)  # (N, K, K)
        triu_mask = torch.triu(
            torch.ones((K, K), dtype=torch.bool, device=s_mix.device), diagonal=1
        )
        off_diag = gram[:, triu_mask]
        penalty = penalty + alpha1 * torch.mean(off_diag * off_diag)

    if alpha2 > 0.0 and T > 1:
        grad_t = s_mix[..., 1:] - s_mix[..., :-1]
        smooth = torch.mean(grad_t * grad_t)
        penalty = penalty + alpha2 * smooth

    return penalty
