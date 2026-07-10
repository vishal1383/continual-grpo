"""Proposal-specific skill-isolating spectral gradient transformation.

This is intentionally not standard Muon.  A matrix gradient is first removed
from learned protected row/column subspaces, then restricted to its dominant
target-task singular directions, and finally projected again for safety.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ProtectedAxes:
    left: torch.Tensor | None = None   # [rows, protected_rank]
    right: torch.Tensor | None = None  # [cols, protected_rank]


def _remove_left(x: torch.Tensor, basis: torch.Tensor | None) -> torch.Tensor:
    if basis is None or basis.numel() == 0:
        return x
    basis = basis.to(device=x.device, dtype=x.dtype)
    return x - basis @ (basis.mT @ x)


def _remove_right(x: torch.Tensor, basis: torch.Tensor | None) -> torch.Tensor:
    if basis is None or basis.numel() == 0:
        return x
    basis = basis.to(device=x.device, dtype=x.dtype)
    return x - (x @ basis) @ basis.mT


@torch.no_grad()
def isolate_matrix_gradient(
    gradient: torch.Tensor,
    axes: ProtectedAxes | None,
    target_rank: int = 1,
) -> torch.Tensor:
    """Return Q_L [top-k SVD(Q_L G Q_R)] Q_R.

    Vectors/scalars pass through because they have no matrix skill axes.
    SVD is computed in fp32 for numerical stability and returned in the
    original dtype.
    """
    if gradient.ndim != 2 or target_rank <= 0:
        return gradient
    original_dtype = gradient.dtype
    x = gradient.float()
    axes = axes or ProtectedAxes()
    x = _remove_right(_remove_left(x, axes.left), axes.right)
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    k = min(target_rank, s.numel())
    allowed = (u[:, :k] * s[:k]) @ vh[:k]
    allowed = _remove_right(_remove_left(allowed, axes.left), axes.right)
    return allowed.to(original_dtype)


@torch.no_grad()
def transform_gradients(model, axes_by_parameter: dict[str, ProtectedAxes], target_rank: int = 1) -> None:
    """Transform gradients in place immediately before optimizer.step()."""
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            parameter.grad.copy_(isolate_matrix_gradient(
                parameter.grad, axes_by_parameter.get(name), target_rank
            ))
