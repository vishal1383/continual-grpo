import torch

from continual_grpo.contrastive_opsd import contrastive_opsd_loss
from continual_grpo.losses import (clipped_grpo_loss, divergence_masks,
                                   group_advantages, select_pairs)


def test_select_pairs_matches_within_groups():
    positives, negatives = select_pairs([1.0, 0.0, 0.0, 1.0], 2)
    assert positives == [0, 3]
    assert negatives == [1, 2]


def test_select_pairs_skips_uniform_groups():
    assert select_pairs([0.0, 0.0], 2) == ([], [])
    assert select_pairs([1.0, 1.0], 2) == ([], [])


def test_divergence_masks_start_at_first_difference():
    pos_ids = torch.tensor([[5, 6, 7, 8]])
    neg_ids = torch.tensor([[5, 6, 9, 8]])
    pos_mask = torch.tensor([[1, 1, 1, 0]])
    out = divergence_masks(pos_ids, neg_ids, pos_mask)
    assert out.tolist() == [[0.0, 0.0, 1.0, 0.0]]


def test_group_advantages_zero_for_uniform_groups():
    advantages, mixed = group_advantages(torch.tensor([[1.0, 1.0], [1.0, 0.0]]))
    assert advantages[0].abs().max() == 0.0
    assert advantages[1].sum().abs() < 1e-6
    assert mixed.tolist() == [False, True]


def test_clipped_grpo_loss_zero_at_on_policy_point():
    policy = torch.tensor([[-1.0, -2.0], [-1.5, -0.5]])
    adv = torch.tensor([[1.0], [-1.0]])
    mask = torch.ones(2, 2)
    lengths = mask.sum(1)
    loss = clipped_grpo_loss(policy, policy.detach(), adv, mask, lengths, 0.2)
    assert float(loss.detach()) == 0.0
    assert torch.isfinite(loss)


def test_contrastive_opsd_loss_is_finite():
    torch.manual_seed(0)
    pairs, length, vocab = 2, 4, 11
    loss, stats = contrastive_opsd_loss(
        torch.randn(pairs, length, vocab), torch.randn(pairs, length, vocab),
        torch.randn(pairs, length), torch.randn(pairs, length),
        torch.ones(pairs, length), torch.ones(pairs, length), torch.ones(pairs, length))
    assert torch.isfinite(loss)
    assert set(stats) == {"opsd_positive_kl", "opsd_contrastive"}
