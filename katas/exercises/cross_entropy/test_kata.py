"""Tests for cross-entropy loss kata."""

import torch
import torch.nn.functional as F


try:
    from user_kata import cross_entropy_loss
    from user_kata import binary_cross_entropy
    from user_kata import cross_entropy_with_label_smoothing
except ImportError:
    from .reference import cross_entropy_loss
    from .reference import binary_cross_entropy
    from .reference import cross_entropy_with_label_smoothing


def test_cross_entropy_basic():

    logits = torch.randn(10, 5)
    targets = torch.randint(0, 5, (10,))

    result = cross_entropy_loss(logits, targets)
    expected = F.cross_entropy(logits, targets)

    assert torch.allclose(result, expected, atol=1e-5)


def test_cross_entropy_perfect_prediction():

    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    targets = torch.tensor([0, 1])

    result = cross_entropy_loss(logits, targets)
    assert result < 0.1  # Should be very small


def test_binary_cross_entropy_basic():

    predictions = torch.sigmoid(torch.randn(20))
    targets = torch.randint(0, 2, (20,)).float()

    result = binary_cross_entropy(predictions, targets)
    expected = F.binary_cross_entropy(predictions, targets)

    assert torch.allclose(result, expected, atol=1e-5)


def test_binary_cross_entropy_perfect():

    predictions = torch.tensor([1.0, 0.0, 1.0, 0.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    result = binary_cross_entropy(predictions, targets)
    assert result < 0.01  # Should be very small


def test_label_smoothing():

    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))

    result = cross_entropy_with_label_smoothing(logits, targets, smoothing=0.1)

    # Loss with smoothing should be close to but different from standard CE
    standard_ce = F.cross_entropy(logits, targets)
    assert not torch.allclose(result, standard_ce)
    assert result.item() > 0


def test_label_smoothing_no_smoothing():

    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))

    result = cross_entropy_with_label_smoothing(logits, targets, smoothing=0.0)
    expected = F.cross_entropy(logits, targets)

    assert torch.allclose(result, expected, atol=1e-5)
