"""Tests for get max attention positions kata."""

import pytest
import torch
from framework import assert_shape

try:
    from user_kata import get_max_attention_positions
except ImportError:
    from .reference import get_max_attention_positions


def test_output_shape():
    """Output should have shape (batch, n_heads, seq_q, top_k)"""
    patterns = torch.rand(2, 4, 10, 15)
    top_k = 3

    indices = get_max_attention_positions(patterns, top_k=top_k)
    assert_shape(indices, (2, 4, 10, 3))


def test_indices_in_range():
    """Indices should be within valid range [0, seq_k)"""
    patterns = torch.rand(1, 2, 5, 8)
    indices = get_max_attention_positions(patterns, top_k=3)

    assert (indices >= 0).all()
    assert (indices < 8).all()  # seq_k = 8


def test_correctness_simple():
    """Verify correctness on simple known pattern"""
    # Create pattern where we know the top positions
    patterns = torch.zeros(1, 1, 1, 5)
    patterns[0, 0, 0, :] = torch.tensor([0.1, 0.5, 0.2, 0.05, 0.15])

    top_3 = get_max_attention_positions(patterns, top_k=3)

    # Top 3 should be: position 1 (0.5), position 2 (0.2), position 4 (0.15)
    expected = torch.tensor([[[1, 2, 4]]])
    assert torch.equal(top_3, expected)


def test_descending_order():
    """Indices should be ordered by attention weight (descending)"""
    patterns = torch.tensor([[[[0.05, 0.3, 0.5, 0.1, 0.05]]]])  # (1,1,1,5)

    top_3 = get_max_attention_positions(patterns, top_k=3)

    # Should be [2, 1, 3] (0.5, 0.3, 0.1)
    expected = torch.tensor([[[2, 1, 3]]])
    assert torch.equal(top_3, expected)


def test_variable_top_k():
    """Should respect different top_k values"""
    patterns = torch.rand(1, 1, 3, 10)

    top_1 = get_max_attention_positions(patterns, top_k=1)
    top_5 = get_max_attention_positions(patterns, top_k=5)

    assert_shape(top_1, (1, 1, 3, 1))
    assert_shape(top_5, (1, 1, 3, 5))


def test_multiple_queries():
    """Each query should have independent top-k"""
    patterns = torch.zeros(1, 1, 3, 4)
    patterns[0, 0, 0, :] = torch.tensor([0.4, 0.3, 0.2, 0.1])  # Query 0: top is 0
    patterns[0, 0, 1, :] = torch.tensor([0.1, 0.4, 0.3, 0.2])  # Query 1: top is 1
    patterns[0, 0, 2, :] = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Query 2: top is 3

    top_1 = get_max_attention_positions(patterns, top_k=1)

    assert top_1[0, 0, 0, 0] == 0  # Query 0's top position
    assert top_1[0, 0, 1, 0] == 1  # Query 1's top position
    assert top_1[0, 0, 2, 0] == 3  # Query 2's top position


def test_multiple_heads():
    """Each head should have independent top-k"""
    patterns = torch.zeros(1, 2, 1, 3)
    patterns[0, 0, 0, :] = torch.tensor([0.5, 0.3, 0.2])  # Head 0: top is 0
    patterns[0, 1, 0, :] = torch.tensor([0.2, 0.3, 0.5])  # Head 1: top is 2

    top_1 = get_max_attention_positions(patterns, top_k=1)

    assert top_1[0, 0, 0, 0] == 0  # Head 0's top position
    assert top_1[0, 1, 0, 0] == 2  # Head 1's top position


def test_batch_independence():
    """Each batch element should have independent top-k"""
    patterns = torch.zeros(2, 1, 1, 3)
    patterns[0, 0, 0, :] = torch.tensor([0.5, 0.3, 0.2])  # Batch 0: top is 0
    patterns[1, 0, 0, :] = torch.tensor([0.2, 0.5, 0.3])  # Batch 1: top is 1

    top_1 = get_max_attention_positions(patterns, top_k=1)

    assert top_1[0, 0, 0, 0] == 0  # Batch 0's top position
    assert top_1[1, 0, 0, 0] == 1  # Batch 1's top position


def test_ties_handled_consistently():
    """Ties should be handled consistently (any valid order acceptable)"""
    patterns = torch.tensor([[[[0.5, 0.5, 0.0]]]])  # Tie between positions 0 and 1

    top_2 = get_max_attention_positions(patterns, top_k=2)

    # Should include both positions with 0.5 (order may vary)
    top_positions = set(top_2[0, 0, 0].tolist())
    assert top_positions == {0, 1}
