import torch

from continual_grpo.contrastive_opsd import contrastive_opsd_loss
from continual_grpo.opsd_trainer import divergence_masks, select_pairs


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


def test_contrastive_opsd_loss_is_finite():
    torch.manual_seed(0)
    pairs, length, vocab = 2, 4, 11
    loss, stats = contrastive_opsd_loss(
        torch.randn(pairs, length, vocab), torch.randn(pairs, length, vocab),
        torch.randn(pairs, length), torch.randn(pairs, length),
        torch.ones(pairs, length), torch.ones(pairs, length), torch.ones(pairs, length))
    assert torch.isfinite(loss)
    assert set(stats) == {"opsd_positive_kl", "opsd_contrastive"}
