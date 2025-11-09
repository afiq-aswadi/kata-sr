"""Tests for applying attention to values kata."""

import torch

from framework import assert_close, assert_shape

try:
    from user_kata import apply_attention_to_values
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    apply_attention_to_values = reference.apply_attention_to_values  # type: ignore


def test_output_shape():
    """Output should have shape (batch, seq_q, d_model)."""
    batch, seq_q, seq_k, d_model = 2, 5, 7, 64
    weights = torch.randn(batch, seq_q, seq_k)
    V = torch.randn(batch, seq_k, d_model)

    output = apply_attention_to_values(weights, V)
    assert_shape(output, (batch, seq_q, d_model), "attention output")


def test_self_attention_shape():
    """Self-attention should maintain sequence length."""
    batch, seq_len, d_model = 2, 10, 32
    weights = torch.randn(batch, seq_len, seq_len)
    V = torch.randn(batch, seq_len, d_model)

    output = apply_attention_to_values(weights, V)
    assert_shape(output, (batch, seq_len, d_model), "self-attention output")


def test_uniform_weights():
    """Uniform weights should produce average of values."""
    batch, seq_len, d_model = 1, 3, 4
    weights = torch.ones(batch, seq_len, seq_len) / seq_len  # Uniform weights
    V = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]]])

    output = apply_attention_to_values(weights, V)

    # Average of values
    expected = torch.tensor([[[5.0, 6.0, 7.0, 8.0],
                              [5.0, 6.0, 7.0, 8.0],
                              [5.0, 6.0, 7.0, 8.0]]])

    assert_close(output, expected, name="uniform attention output")


def test_one_hot_weights():
    """One-hot weights should select specific value vector."""
    batch, seq_len, d_model = 1, 4, 8
    V = torch.randn(batch, seq_len, d_model)

    for i in range(seq_len):
        # One-hot weights: attend only to position i
        weights = torch.zeros(batch, seq_len, seq_len)
        weights[:, :, i] = 1.0

        output = apply_attention_to_values(weights, V)

        # All positions should have the same output (V[i])
        expected = V[:, i:i+1, :].expand(batch, seq_len, d_model)
        assert_close(output, expected, name=f"one-hot attention to position {i}")


def test_different_dimensions():
    """Should work with different d_model dimensions."""
    batch, seq_len = 2, 5

    for d_model in [8, 32, 64, 128]:
        weights = torch.randn(batch, seq_len, seq_len)
        V = torch.randn(batch, seq_len, d_model)
        output = apply_attention_to_values(weights, V)
        assert_shape(output, (batch, seq_len, d_model), f"d_model={d_model}")


def test_cross_attention():
    """Should handle different sequence lengths (cross-attention)."""
    batch, seq_q, seq_k, d_model = 2, 3, 5, 32
    weights = torch.randn(batch, seq_q, seq_k)
    V = torch.randn(batch, seq_k, d_model)

    output = apply_attention_to_values(weights, V)
    assert_shape(output, (batch, seq_q, d_model), "cross-attention output")


def test_no_nans_or_infs():
    """Output should not contain NaNs or infinities."""
    batch, seq_len, d_model = 2, 5, 64
    weights = torch.randn(batch, seq_len, seq_len)
    V = torch.randn(batch, seq_len, d_model)

    output = apply_attention_to_values(weights, V)

    assert not torch.isnan(output).any(), "output should not contain NaN"
    assert not torch.isinf(output).any(), "output should not contain inf"
