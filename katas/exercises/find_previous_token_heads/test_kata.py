"""Tests for find_previous_token_heads kata."""

import pytest
import torch

try:
    from user_kata import find_previous_token_heads
except ImportError:
    from .reference import find_previous_token_heads


def test_output_shape():
    """Output should be boolean tensor of shape (n_heads,)."""
    patterns = torch.rand(2, 5, 8, 8)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    result = find_previous_token_heads(patterns)

    assert result.shape == (5,)
    assert result.dtype == torch.bool


def test_previous_token_head_detected():
    """Should detect head that attends to previous token."""
    patterns = torch.zeros(1, 3, 6, 6)

    # Head 0: attends to previous token (diagonal - 1)
    for i in range(1, 6):
        patterns[0, 0, i, i - 1] = 0.9
        patterns[0, 0, i, i] = 0.1

    # Head 1: attends to first token
    patterns[0, 1, :, 0] = 1.0

    # Head 2: uniform
    patterns[0, 2, :, :] = 1.0 / 6

    result = find_previous_token_heads(patterns, threshold=0.5)

    assert result[0] == True   # Previous-token head
    assert result[1] == False  # First-token head
    assert result[2] == False  # Uniform


def test_threshold_sensitivity():
    """Different thresholds should give different results."""
    patterns = torch.zeros(1, 2, 5, 5)

    # Head 0: 0.6 average to previous token
    for i in range(1, 5):
        patterns[0, 0, i, i - 1] = 0.6
        patterns[0, 0, i, i] = 0.4

    # Head 1: 0.3 average to previous token
    for i in range(1, 5):
        patterns[0, 1, i, i - 1] = 0.3
        patterns[0, 1, i, i] = 0.7

    result_low = find_previous_token_heads(patterns, threshold=0.2)
    result_high = find_previous_token_heads(patterns, threshold=0.5)

    assert result_low[0] == True and result_low[1] == True   # Both pass low threshold
    assert result_high[0] == True and result_high[1] == False  # Only head 0 passes high


def test_batch_averaging():
    """Should average across batches correctly."""
    patterns = torch.zeros(3, 1, 4, 4)

    # First batch: strong previous-token attention
    for i in range(1, 4):
        patterns[0, 0, i, i - 1] = 0.9

    # Second batch: weak previous-token attention
    for i in range(1, 4):
        patterns[1, 0, i, i - 1] = 0.2

    # Third batch: medium previous-token attention
    for i in range(1, 4):
        patterns[2, 0, i, i - 1] = 0.5

    # Normalize
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    # Average should be (0.9 + 0.2 + 0.5) / 3 ≈ 0.53
    result = find_previous_token_heads(patterns, threshold=0.5)

    assert result[0] == True


def test_no_previous_token_heads():
    """Should return all False when no heads qualify."""
    patterns = torch.zeros(1, 4, 5, 5)

    # All heads attend uniformly (or to first token)
    patterns[:, :, :, 0] = 1.0

    result = find_previous_token_heads(patterns, threshold=0.4)

    assert not result.any()  # All False


def test_all_previous_token_heads():
    """Should return all True when all heads qualify."""
    patterns = torch.zeros(1, 3, 7, 7)

    # All heads attend strongly to previous token
    for head in range(3):
        for i in range(1, 7):
            patterns[0, head, i, i - 1] = 0.95

    # Normalize
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    result = find_previous_token_heads(patterns, threshold=0.4)

    assert result.all()  # All True


def test_short_sequence():
    """Handle short sequences (only 2-3 tokens)."""
    patterns = torch.zeros(1, 2, 3, 3)

    # Head 0: attends to previous
    patterns[0, 0, 1, 0] = 0.8
    patterns[0, 0, 2, 1] = 0.8

    # Normalize
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    result = find_previous_token_heads(patterns, threshold=0.5)

    assert result[0] == True
