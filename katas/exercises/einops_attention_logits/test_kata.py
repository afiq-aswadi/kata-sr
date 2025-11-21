"""Tests for scaled_attention_logits."""

import math

import pytest
import torch

try:
    from user_kata import scaled_attention_logits
except ImportError:
    from .reference import scaled_attention_logits


def _baseline_logits(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    return torch.einsum("bhqd,bhkd->bhqk", query, key) * scale


def test_logits_match_baseline():
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # b=1,h=1,q=2,d=2
    key = torch.tensor([[[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]])  # k=3

    result = scaled_attention_logits(query, key)
    expected = _baseline_logits(query, key)
    assert torch.allclose(result, expected)
    assert result.shape == (1, 1, 2, 3)


def test_random_logits_align_with_torch():
    torch.manual_seed(0)
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 4, 6, 8)
    expected = _baseline_logits(query, key)
    result = scaled_attention_logits(query, key)
    assert torch.allclose(result, expected, atol=1e-6)


def test_mismatched_dims_raise():
    q = torch.zeros(1, 2, 3, 4)
    k = torch.zeros(1, 3, 4, 4)
    with pytest.raises(AssertionError):
        _ = scaled_attention_logits(q, k)
