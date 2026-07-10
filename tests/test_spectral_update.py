import torch

from continual_grpo.spectral_update import ProtectedAxes, isolate_matrix_gradient


def test_update_is_rank_limited_and_protected():
    torch.manual_seed(7)
    gradient = torch.randn(6, 5)
    left, _ = torch.linalg.qr(torch.randn(6, 2))
    right, _ = torch.linalg.qr(torch.randn(5, 2))
    update = isolate_matrix_gradient(gradient, ProtectedAxes(left, right), target_rank=1)
    assert torch.linalg.matrix_rank(update, tol=1e-5) <= 1
    assert torch.max(torch.abs(left.mT @ update)) < 1e-5
    assert torch.max(torch.abs(update @ right)) < 1e-5


def test_non_matrix_gradients_pass_through():
    gradient = torch.randn(8)
    update = isolate_matrix_gradient(gradient, ProtectedAxes(), target_rank=1)
    assert torch.equal(update, gradient)
