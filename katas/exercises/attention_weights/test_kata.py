"""Tests for attention weights kata."""

import torch

from framework import assert_close, assert_shape

try:
    from user_kata import compute_attention_weights
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    compute_attention_weights = reference.compute_attention_weights  # type: ignore


def test_output_shape():
    """Output should have same shape as input scores."""
    batch, seq_q, seq_k = 2, 5, 7
    scores = torch.randn(batch, seq_q, seq_k)

    weights = compute_attention_weights(scores)
    assert_shape(weights, (batch, seq_q, seq_k), "attention weights")


def test_probabilities_sum_to_one():
    """Each row of attention weights should sum to 1."""
    batch, seq_len = 2, 10
    scores = torch.randn(batch, seq_len, seq_len)

    weights = compute_attention_weights(scores)
    row_sums = weights.sum(dim=-1)

    expected = torch.ones(batch, seq_len)
    assert_close(row_sums, expected, atol=1e-6, name="probability sums")


def test_no_mask():
    """Without mask, should apply standard softmax."""
    batch, seq_len = 2, 5
    scores = torch.randn(batch, seq_len, seq_len)

    weights = compute_attention_weights(scores, mask=None)
    expected = torch.softmax(scores, dim=-1)

    assert_close(weights, expected, name="softmax without mask")


def test_causal_mask():
    """Causal mask should zero out future positions."""
    seq_len = 4
    scores = torch.ones(1, seq_len, seq_len)

    # Upper triangular mask (mask future positions)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    weights = compute_attention_weights(scores, mask=mask)

    # First position should only attend to itself (weight=1.0)
    assert_close(weights[0, 0, 0], torch.tensor(1.0), name="first position weight")
    assert_close(weights[0, 0, 1:], torch.zeros(seq_len - 1), name="masked positions")


def test_mask_prevents_attention():
    """Masked positions should have zero attention weight."""
    batch, seq_len = 2, 5
    scores = torch.randn(batch, seq_len, seq_len)

    # Mask out last 2 positions
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    mask[:, -2:] = True

    weights = compute_attention_weights(scores, mask=mask)

    # Last 2 positions should have zero weight
    assert_close(weights[:, :, -2:], torch.zeros(batch, seq_len, 2), name="masked weights")


def test_different_sequence_lengths():
    """Should handle different sequence lengths."""
    batch = 2

    for seq_len in [1, 3, 10, 20]:
        scores = torch.randn(batch, seq_len, seq_len)
        weights = compute_attention_weights(scores)
        assert_shape(weights, (batch, seq_len, seq_len), f"seq_len={seq_len}")


def test_no_nans_or_infs():
    """Output should not contain NaNs or infinities."""
    batch, seq_len = 2, 5
    scores = torch.randn(batch, seq_len, seq_len)

    weights = compute_attention_weights(scores)

    assert not torch.isnan(weights).any(), "output should not contain NaN"
    assert not torch.isinf(weights).any(), "output should not contain inf"
