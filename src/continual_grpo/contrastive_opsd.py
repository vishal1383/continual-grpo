"""Contrastive on-policy self-distillation objective from the proposal."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_log_likelihood(token_log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(-1).clamp_min(1)
    return (token_log_probs * mask).sum(-1) / lengths


def contrastive_opsd_loss(
    student_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    positive_log_probs: torch.Tensor,
    negative_log_probs: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    divergence_mask: torch.Tensor,
    margin: float = 0.2,
    negative_weight: float = 0.1,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Positive reverse-KL plus bounded correct-vs-wrong sequence margin.

    `divergence_mask` selects branch-point tokens where verified positive and
    negative traces differ, avoiding uniform distillation over every token.
    """
    student = F.log_softmax(student_logits / temperature, dim=-1)
    reference = F.softmax(reference_logits.detach() / temperature, dim=-1)
    token_kl = F.kl_div(student, reference, reduction="none").sum(-1)
    positive = (token_kl * divergence_mask).sum() / divergence_mask.sum().clamp_min(1)
    pos_score = sequence_log_likelihood(positive_log_probs, positive_mask)
    neg_score = sequence_log_likelihood(negative_log_probs, negative_mask)
    contrastive = F.relu(margin - (pos_score - neg_score)).mean()
    total = positive + negative_weight * contrastive
    return total, {"opsd_positive_kl": positive.detach(), "opsd_contrastive": contrastive.detach()}
