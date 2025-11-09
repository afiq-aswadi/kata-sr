"""Tests for compute attention entropy kata."""

import pytest
import torch
from framework import assert_shape

try:
    from user_kata import compute_attention_entropy
except ImportError:
    from .reference import compute_attention_entropy


def test_output_shape():
    """Entropy should have shape (batch, n_heads, seq_q)"""
    patterns = torch.rand(2, 4, 10, 15)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)  # Normalize

    entropy = compute_attention_entropy(patterns)
    assert_shape(entropy, (2, 4, 10))


def test_focused_attention_low_entropy():
    """Focused attention (all weight on one token) should have low entropy"""
    # All attention on first token
    focused = torch.zeros(1, 1, 3, 4)
    focused[:, :, :, 0] = 1.0

    entropy = compute_attention_entropy(focused)

    # Entropy should be near zero (very focused)
    assert (entropy < 0.1).all()


def test_uniform_attention_high_entropy():
    """Uniform attention should have high entropy"""
    # Uniform distribution
    uniform = torch.ones(1, 1, 3, 4) * 0.25

    entropy = compute_attention_entropy(uniform)

    # Entropy for uniform distribution over 4 items is log(4) ≈ 1.386
    expected_entropy = torch.log(torch.tensor(4.0))
    assert torch.allclose(entropy, expected_entropy.expand_as(entropy), atol=0.1)


def test_focused_vs_diffuse():
    """Focused attention should have lower entropy than diffuse"""
    # Focused: 80% on one token
    focused = torch.tensor([[[[0.8, 0.1, 0.05, 0.05]]]])

    # Diffuse: spread evenly
    diffuse = torch.tensor([[[[0.25, 0.25, 0.25, 0.25]]]])

    entropy_focused = compute_attention_entropy(focused)
    entropy_diffuse = compute_attention_entropy(diffuse)

    assert (entropy_focused < entropy_diffuse).all()


def test_entropy_non_negative():
    """Entropy should always be non-negative"""
    patterns = torch.rand(4, 8, 20, 20)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    entropy = compute_attention_entropy(patterns)

    assert (entropy >= 0).all()


def test_entropy_finite():
    """Entropy should be finite (no NaN or Inf)"""
    patterns = torch.rand(2, 4, 10, 10)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    entropy = compute_attention_entropy(patterns)

    assert torch.isfinite(entropy).all()


def test_max_entropy_bound():
    """Entropy should be bounded by log(seq_len)"""
    seq_len = 16
    patterns = torch.rand(1, 2, 5, seq_len)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    entropy = compute_attention_entropy(patterns)
    max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float32))

    # Add small tolerance for numerical errors
    assert (entropy <= max_entropy + 0.1).all()


def test_batch_independence():
    """Entropy computed independently for each batch element"""
    # Create patterns where first batch is focused, second is uniform
    patterns = torch.zeros(2, 1, 1, 4)
    patterns[0, :, :, 0] = 1.0  # Focused
    patterns[1, :, :, :] = 0.25  # Uniform

    entropy = compute_attention_entropy(patterns)

    # First batch should have lower entropy
    assert entropy[0] < entropy[1]
