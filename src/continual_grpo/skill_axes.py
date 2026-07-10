"""Build protected skill axes from anchor-batch gradients."""
from __future__ import annotations

import torch

from .spectral_update import ProtectedAxes


@torch.no_grad()
def axes_from_gradient_samples(
    samples: dict[str, list[torch.Tensor]], protected_rank: int = 2
) -> dict[str, ProtectedAxes]:
    """Extract row/column bases from accumulated protected-skill gradients.

    Each sample is a matrix gradient from an MMLU/BBQ/bias anchor loss.  The
    bases represent directions whose movement should be suppressed later.
    """
    result: dict[str, ProtectedAxes] = {}
    for name, gradients in samples.items():
        matrices = [g.detach().float().cpu() for g in gradients if g.ndim == 2]
        if not matrices:
            continue
        mean_gradient = torch.stack(matrices).mean(0)
        u, _, vh = torch.linalg.svd(mean_gradient, full_matrices=False)
        rank = min(protected_rank, u.shape[1], vh.shape[0])
        result[name] = ProtectedAxes(left=u[:, :rank], right=vh[:rank].mT)
    return result


def collect_gradient_sample(model, destination: dict[str, list[torch.Tensor]]) -> None:
    """Copy the current anchor-loss gradients into a sample collection."""
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and parameter.grad.ndim == 2:
            destination.setdefault(name, []).append(parameter.grad.detach().cpu())
