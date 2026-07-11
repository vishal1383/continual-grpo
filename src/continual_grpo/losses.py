"""Shared loss pieces for the handwritten GRPO ablations.

Every function receives logits that were already divided by the rollout
temperature, so ratios, KLs, and sequence scores describe the same tempered
policy that actually sampled the completions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .contrastive_opsd import contrastive_opsd_loss


def token_logps(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float(), -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def group_advantages(rewards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-normalized advantages; uniform-reward groups contribute zero."""
    mean = rewards.mean(1, keepdim=True)
    std = rewards.std(1, keepdim=True, unbiased=False)
    advantages = torch.where(std > 1e-8, (rewards - mean) / std.clamp_min(1e-8), 0.0)
    return advantages, (std.squeeze(1) > 1e-8)


def select_pairs(rewards, group_size: int, sequence_scores=None) -> tuple[list[int], list[int]]:
    """Pair each positive with the highest-likelihood negative in its group."""
    values = [float(v) for v in rewards]
    scores = [float(v) for v in sequence_scores] if sequence_scores is not None else None
    positives, negatives = [], []
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        pos = [start + i for i, v in enumerate(group) if v >= 1.0]
        neg = [start + i for i, v in enumerate(group) if v < 1.0]
        if pos and neg:
            chosen = max(neg, key=lambda index: scores[index]) if scores is not None else neg[0]
            for index in pos:
                positives.append(index)
                negatives.append(chosen)
    return positives, negatives


def divergence_masks(positive_ids: torch.Tensor, negative_ids: torch.Tensor,
                     positive_mask: torch.Tensor) -> torch.Tensor:
    """Mask each paired completion from the first token where it differs from
    its negative counterpart onward."""
    out = torch.zeros_like(positive_mask)
    for row in range(positive_ids.shape[0]):
        different = ((positive_ids[row] != negative_ids[row]) & positive_mask[row].bool()).nonzero()
        start = int(different[0]) if different.numel() else 0
        out[row, start:] = positive_mask[row, start:]
    return out


def clipped_grpo_loss(policy_logps: torch.Tensor, old_logps: torch.Tensor,
                      advantages: torch.Tensor, mask: torch.Tensor,
                      lengths: torch.Tensor, clip_eps: float) -> torch.Tensor:
    ratio = torch.exp(policy_logps - old_logps)
    pg = torch.minimum(ratio * advantages, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages)
    return -((pg * mask).sum(1) / lengths).mean()


def reference_kl_loss(policy_logps: torch.Tensor, reference_logps: torch.Tensor,
                      mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    log_ratio = reference_logps - policy_logps
    per_token = torch.exp(log_ratio) - log_ratio - 1
    return ((per_token * mask).sum(1) / lengths).mean()


def opsd_loss_for_chunk(correct: torch.Tensor, group_size: int, completion_ids: torch.Tensor,
                        completion_logits: torch.Tensor, reference_logits: torch.Tensor,
                        policy_logps: torch.Tensor, mask: torch.Tensor,
                        margin: float, negative_weight: float,
                        temperature: float = 1.0) -> tuple[torch.Tensor, int]:
    """C-OPSD over the correct/incorrect pairs of one whole-group chunk."""
    lengths = mask.sum(1).clamp_min(1)
    sequence_scores = (policy_logps * mask).sum(1) / lengths
    positives, negatives = select_pairs(correct.flatten(), group_size, sequence_scores.detach())
    if not positives:
        return torch.zeros((), device=policy_logps.device), 0
    pos = torch.tensor(positives, device=policy_logps.device)
    neg = torch.tensor(negatives, device=policy_logps.device)
    divergence = divergence_masks(completion_ids[pos], completion_ids[neg], mask[pos])
    loss, _ = contrastive_opsd_loss(
        completion_logits[pos].float(), reference_logits[pos].float(),
        policy_logps[pos], policy_logps[neg], mask[pos], mask[neg], divergence,
        margin=margin, negative_weight=negative_weight, temperature=temperature,
    )
    return loss, len(positives)
