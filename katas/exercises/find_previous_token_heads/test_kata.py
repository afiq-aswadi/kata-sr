"""Tests for find previous token heads kata."""

import pytest
import torch
from framework import assert_shape

try:
    from user_kata import find_previous_token_heads
except ImportError:
    from .reference import find_previous_token_heads


def test_output_shape():
    """Output should be boolean tensor of shape (n_heads,)"""
    patterns = torch.rand(2, 8, 10, 10)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    prev_heads = find_previous_token_heads(patterns)
    assert_shape(prev_heads, (8,))
    assert prev_heads.dtype == torch.bool


def test_detect_previous_token_head():
    """Should detect head that attends to previous token"""
    batch, n_heads, seq = 1, 3, 5
    patterns = torch.zeros(batch, n_heads, seq, seq)

    # Head 0: strong previous token attention (90%)
    for i in range(1, seq):
        patterns[0, 0, i, i-1] = 0.9
        patterns[0, 0, i, i] = 0.1

    # Head 1: uniform attention
    patterns[0, 1, :, :] = 1.0 / seq

    # Head 2: self attention
    for i in range(seq):
        patterns[0, 2, i, i] = 1.0

    prev_heads = find_previous_token_heads(patterns, threshold=0.5)

    assert prev_heads[0] == True   # Head 0 should be detected
    assert prev_heads[1] == False  # Head 1 should not
    assert prev_heads[2] == False  # Head 2 should not


def test_threshold_sensitivity():
    """Should respect threshold parameter"""
    patterns = torch.zeros(1, 2, 4, 4)

    # Head 0: 60% to previous token
    for i in range(1, 4):
        patterns[0, 0, i, i-1] = 0.6
        patterns[0, 0, i, i] = 0.4

    # Head 1: 40% to previous token
    for i in range(1, 4):
        patterns[0, 1, i, i-1] = 0.4
        patterns[0, 1, i, i] = 0.6

    # With threshold 0.5, only head 0 should qualify
    prev_heads_high = find_previous_token_heads(patterns, threshold=0.5)
    assert prev_heads_high[0] == True
    assert prev_heads_high[1] == False

    # With threshold 0.3, both should qualify
    prev_heads_low = find_previous_token_heads(patterns, threshold=0.3)
    assert prev_heads_low[0] == True
    assert prev_heads_low[1] == True


def test_single_token_edge_case():
    """Should handle single token gracefully"""
    patterns = torch.ones(1, 4, 1, 1)
    prev_heads = find_previous_token_heads(patterns)

    # With single token, no previous token exists
    assert_shape(prev_heads, (4,))
    assert not prev_heads.any()  # All should be False


def test_batch_averaging():
    """Should average across batch dimension"""
    patterns = torch.zeros(2, 1, 3, 3)

    # Batch 0: strong previous token attention
    for i in range(1, 3):
        patterns[0, 0, i, i-1] = 0.8

    # Batch 1: weak previous token attention
    for i in range(1, 3):
        patterns[1, 0, i, i-1] = 0.3

    # Average is 0.55, should exceed threshold 0.5
    prev_heads = find_previous_token_heads(patterns, threshold=0.5)
    assert prev_heads[0] == True


def test_position_averaging():
    """Should average across sequence positions"""
    patterns = torch.zeros(1, 1, 4, 4)

    # Position 1: strong previous token attention
    patterns[0, 0, 1, 0] = 0.9

    # Position 2: weak previous token attention
    patterns[0, 0, 2, 1] = 0.3

    # Position 3: medium previous token attention
    patterns[0, 0, 3, 2] = 0.6

    # Average is (0.9 + 0.3 + 0.6) / 3 = 0.6
    prev_heads = find_previous_token_heads(patterns, threshold=0.5)
    assert prev_heads[0] == True


def test_no_heads_qualify():
    """Should handle case where no heads qualify"""
    patterns = torch.zeros(1, 3, 5, 5)

    # All heads attend to self, not previous
    for i in range(5):
        patterns[0, :, i, i] = 1.0

    prev_heads = find_previous_token_heads(patterns, threshold=0.5)
    assert not prev_heads.any()


def test_all_heads_qualify():
    """Should handle case where all heads qualify"""
    patterns = torch.zeros(1, 3, 5, 5)

    # All heads attend strongly to previous token
    for i in range(1, 5):
        patterns[0, :, i, i-1] = 0.9

    prev_heads = find_previous_token_heads(patterns, threshold=0.5)
    assert prev_heads.all()
