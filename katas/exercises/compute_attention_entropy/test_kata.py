"""Tests for compute_attention_entropy kata."""

import pytest
import torch

try:
    from user_kata import compute_attention_entropy
except ImportError:
    from .reference import compute_attention_entropy


def test_output_shape():
    """Entropy should have shape (batch, n_heads, query_pos)."""
    patterns = torch.rand(2, 3, 5, 7)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)  # Normalize

    entropy = compute_attention_entropy(patterns)

    assert entropy.shape == (2, 3, 5)


def test_entropy_positive():
    """Entropy should always be non-negative."""
    patterns = torch.rand(4, 2, 10, 10)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    entropy = compute_attention_entropy(patterns)

    assert (entropy >= 0).all()


def test_uniform_has_high_entropy():
    """Uniform distribution should have higher entropy than focused."""
    # Uniform attention
    uniform = torch.ones(1, 1, 5, 10) / 10
    uniform_entropy = compute_attention_entropy(uniform)

    # Focused attention (attending mostly to first token)
    focused = torch.zeros(1, 1, 5, 10)
    focused[:, :, :, 0] = 0.9
    focused[:, :, :, 1:] = 0.1 / 9
    focused_entropy = compute_attention_entropy(focused)

    assert uniform_entropy.mean() > focused_entropy.mean()


def test_deterministic_has_zero_entropy():
    """One-hot (deterministic) attention should have near-zero entropy."""
    # One-hot attention pattern
    patterns = torch.zeros(1, 1, 3, 5)
    patterns[:, :, :, 0] = 1.0  # All queries attend to first key

    entropy = compute_attention_entropy(patterns)

    # Should be very close to 0
    assert entropy.abs().max() < 0.01


def test_batch_independence():
    """Entropy should be computed independently for each batch."""
    # Create two different patterns
    patterns = torch.zeros(2, 1, 3, 4)
    patterns[0, :, :, :] = 1.0 / 4  # Uniform
    patterns[1, :, :, 0] = 1.0       # Deterministic

    entropy = compute_attention_entropy(patterns)

    # First batch should have higher entropy than second
    assert entropy[0].mean() > entropy[1].mean()


def test_head_independence():
    """Entropy computed independently for each head."""
    patterns = torch.zeros(1, 3, 2, 5)
    patterns[0, 0, :, :] = 1.0 / 5   # Head 0: uniform
    patterns[0, 1, :, 0] = 1.0       # Head 1: deterministic
    patterns[0, 2, :, :] = 1.0 / 5   # Head 2: uniform

    entropy = compute_attention_entropy(patterns)

    assert entropy[0, 0].mean() > entropy[0, 1].mean()
    assert entropy[0, 2].mean() > entropy[0, 1].mean()


def test_handles_small_probabilities():
    """Should handle very small probabilities without error."""
    patterns = torch.zeros(1, 1, 2, 10)
    patterns[:, :, :, 0] = 0.999
    patterns[:, :, :, 1:] = 0.001 / 9

    entropy = compute_attention_entropy(patterns)

    # Should not be NaN or inf
    assert not torch.isnan(entropy).any()
    assert not torch.isinf(entropy).any()


def test_normalized_patterns():
    """Works correctly with properly normalized patterns."""
    # Create random patterns and normalize
    patterns = torch.rand(3, 4, 8, 12)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    # Verify they sum to 1.0
    assert torch.allclose(patterns.sum(dim=-1), torch.ones(3, 4, 8), atol=1e-5)

    entropy = compute_attention_entropy(patterns)

    # Should be valid
    assert (entropy >= 0).all()
    assert not torch.isnan(entropy).any()
