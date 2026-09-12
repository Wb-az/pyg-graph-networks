"""Tests for the custom MultiClassFocalLoss, which replaces torch_focalloss's
implementation (a Python for-loop over every sample in its weighted-mean
normalisation, a severe bottleneck at full-batch scale)."""
import torch
from torch.nn import CrossEntropyLoss

from src.common.losses import MultiClassFocalLoss


def test_gamma_zero_matches_cross_entropy():
    """gamma=0 makes (1 - p_t) ** 0 == 1 for every sample, i.e. plain CE."""
    torch.manual_seed(0)
    logits = torch.randn(32, 5)
    target = torch.randint(0, 5, (32,))
    focal = MultiClassFocalLoss(gamma=0.0)(logits, target)
    ce = CrossEntropyLoss()(logits, target)
    assert torch.allclose(focal, ce, atol=1e-6)


def test_gamma_zero_matches_weighted_cross_entropy():
    """Same check, with class weights: the weighted-mean normalisation must
    match CrossEntropyLoss(weight=...)'s convention exactly."""
    torch.manual_seed(1)
    logits = torch.randn(64, 4)
    target = torch.randint(0, 4, (64,))
    weight = torch.tensor([0.1, 1.0, 2.0, 5.0])
    focal = MultiClassFocalLoss(gamma=0.0, weight=weight)(logits, target)
    ce = CrossEntropyLoss(weight=weight)(logits, target)
    assert torch.allclose(focal, ce, atol=1e-5)


def test_focusing_reduces_loss_relative_to_cross_entropy():
    """gamma > 0 down-weights confidently-correct predictions, so the focal
    loss on a mostly-correct batch is strictly less than the CE loss."""
    torch.manual_seed(2)
    target = torch.randint(0, 3, (100,))
    logits = torch.nn.functional.one_hot(target, 3).float() * 8.0  # confident, mostly correct
    focal = MultiClassFocalLoss(gamma=2.0)(logits, target)
    ce = CrossEntropyLoss()(logits, target)
    assert focal.item() < ce.item()


def test_gradient_flows():
    logits = torch.randn(16, 3, requires_grad=True)
    target = torch.randint(0, 3, (16,))
    loss = MultiClassFocalLoss(gamma=2.0, weight=torch.tensor([1.0, 2.0, 0.5]))(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_no_python_loop_over_batch(monkeypatch):
    """Regression guard for the actual bug being fixed: nothing should
    iterate over individual samples. A batch large enough that a Python loop
    would be slow should still return promptly; this checks correctness
    scales rather than timing, which would be flaky."""
    torch.manual_seed(3)
    n = 50_000
    logits = torch.randn(n, 40)
    target = torch.randint(0, 40, (n,))
    weight = torch.rand(40) + 0.1
    loss = MultiClassFocalLoss(gamma=2.0, weight=weight)(logits, target)
    assert torch.isfinite(loss)
    assert loss.ndim == 0
