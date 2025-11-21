"""Tests for split_heads."""

import pytest
import torch

try:
    from user_kata import split_heads
except ImportError:
    from .reference import split_heads


def test_split_heads_rearranges_correctly():
    hidden = torch.arange(1 * 2 * 12, dtype=torch.float32).reshape(1, 2, 12)
    result = split_heads(hidden, num_heads=3)
    assert result.shape == (1, 3, 2, 4)

    expected = hidden.view(1, 2, 3, 4).permute(0, 2, 1, 3)
    assert torch.equal(result, expected)


def test_invalid_head_count_raises():
    hidden = torch.zeros(1, 2, 10)
    with pytest.raises(AssertionError):
        _ = split_heads(hidden, num_heads=3)


def test_multiple_batches_support():
    hidden = torch.randn(4, 6, 32)
    out = split_heads(hidden, num_heads=8)
    assert out.shape == (4, 8, 6, 4)
