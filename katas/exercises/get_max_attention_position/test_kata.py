"""Tests for get_max_attention_position kata."""

import pytest
import torch

try:
    from user_kata import get_max_attention_position
except ImportError:
    from .reference import get_max_attention_position


def test_return_type():
    """Should return tuple of (int, float)."""
    patterns = torch.rand(1, 2, 5, 5)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    pos, weight = get_max_attention_position(patterns, query_pos=2, head=0)

    assert isinstance(pos, int)
    assert isinstance(weight, float)


def test_position_in_range():
    """Returned position should be valid index."""
    patterns = torch.rand(1, 3, 8, 8)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    pos, weight = get_max_attention_position(patterns, query_pos=5, head=1)

    assert 0 <= pos < 8


def test_weight_in_range():
    """Attention weight should be between 0 and 1."""
    patterns = torch.rand(1, 2, 6, 6)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    pos, weight = get_max_attention_position(patterns, query_pos=3, head=0)

    assert 0.0 <= weight <= 1.0


def test_deterministic_pattern():
    """Should find correct max for deterministic pattern."""
    patterns = torch.zeros(1, 1, 4, 4)
    # Query position 2 attends strongly to position 0
    patterns[0, 0, 2, 0] = 0.9
    patterns[0, 0, 2, 1:] = 0.1 / 3

    pos, weight = get_max_attention_position(patterns, query_pos=2, head=0)

    assert pos == 0
    assert abs(weight - 0.9) < 1e-5


def test_uniform_pattern():
    """For uniform pattern, any position is valid (max exists)."""
    patterns = torch.ones(1, 1, 5, 5) / 5

    pos, weight = get_max_attention_position(patterns, query_pos=1, head=0)

    assert 0 <= pos < 5
    assert abs(weight - 0.2) < 1e-5


def test_different_query_positions():
    """Different query positions can have different max positions."""
    patterns = torch.zeros(1, 1, 4, 4)

    # Query 0 attends to position 1
    patterns[0, 0, 0, 1] = 1.0

    # Query 1 attends to position 3
    patterns[0, 0, 1, 3] = 1.0

    # Query 2 attends to position 0
    patterns[0, 0, 2, 0] = 1.0

    pos0, _ = get_max_attention_position(patterns, query_pos=0, head=0)
    pos1, _ = get_max_attention_position(patterns, query_pos=1, head=0)
    pos2, _ = get_max_attention_position(patterns, query_pos=2, head=0)

    assert pos0 == 1
    assert pos1 == 3
    assert pos2 == 0


def test_different_heads():
    """Different heads can have different max positions."""
    patterns = torch.zeros(1, 3, 2, 5)

    # Head 0: attends to position 2
    patterns[0, 0, 0, 2] = 1.0

    # Head 1: attends to position 4
    patterns[0, 1, 0, 4] = 1.0

    # Head 2: attends to position 0
    patterns[0, 2, 0, 0] = 1.0

    pos0, _ = get_max_attention_position(patterns, query_pos=0, head=0)
    pos1, _ = get_max_attention_position(patterns, query_pos=0, head=1)
    pos2, _ = get_max_attention_position(patterns, query_pos=0, head=2)

    assert pos0 == 2
    assert pos1 == 4
    assert pos2 == 0


def test_different_batches():
    """Different batches can have different max positions."""
    patterns = torch.zeros(3, 1, 2, 4)

    # Batch 0: attends to position 1
    patterns[0, 0, 0, 1] = 1.0

    # Batch 1: attends to position 2
    patterns[1, 0, 0, 2] = 1.0

    # Batch 2: attends to position 3
    patterns[2, 0, 0, 3] = 1.0

    pos0, _ = get_max_attention_position(patterns, query_pos=0, head=0, batch_idx=0)
    pos1, _ = get_max_attention_position(patterns, query_pos=0, head=0, batch_idx=1)
    pos2, _ = get_max_attention_position(patterns, query_pos=0, head=0, batch_idx=2)

    assert pos0 == 1
    assert pos1 == 2
    assert pos2 == 3


def test_is_actually_maximum():
    """Returned weight should be the actual maximum."""
    patterns = torch.rand(1, 2, 6, 10)
    patterns = patterns / patterns.sum(dim=-1, keepdim=True)

    query_pos = 3
    head = 1
    pos, weight = get_max_attention_position(patterns, query_pos, head)

    # Verify it's actually the max
    actual_max = patterns[0, head, query_pos, :].max().item()
    assert abs(weight - actual_max) < 1e-6
