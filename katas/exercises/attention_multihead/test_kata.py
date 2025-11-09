"""Tests for multi-head attention kata."""

import torch

from framework import assert_close, assert_shape

try:
    from user_kata import multihead_attention
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    multihead_attention = reference.multihead_attention  # type: ignore


def test_output_shape():
    """Output should have shape (batch, seq_q, d_model)."""
    batch, seq_q, seq_k, d_model = 2, 5, 7, 64
    num_heads = 8

    Q = torch.randn(batch, seq_q, d_model)
    K = torch.randn(batch, seq_k, d_model)
    V = torch.randn(batch, seq_k, d_model)

    output = multihead_attention(Q, K, V, num_heads)
    assert_shape(output, (batch, seq_q, d_model), "multihead attention output")


def test_self_attention():
    """Self-attention should maintain sequence length."""
    batch, seq_len, d_model = 2, 10, 64
    num_heads = 8

    Q = K = V = torch.randn(batch, seq_len, d_model)

    output = multihead_attention(Q, K, V, num_heads)
    assert_shape(output, (batch, seq_len, d_model), "self-attention output")


def test_single_head():
    """Single head should work correctly."""
    batch, seq_len, d_model = 2, 5, 32

    Q = K = V = torch.randn(batch, seq_len, d_model)

    output = multihead_attention(Q, K, V, num_heads=1)
    assert_shape(output, (batch, seq_len, d_model), "single head")


def test_different_num_heads():
    """Should work with different number of heads."""
    batch, seq_len, d_model = 2, 5, 64

    Q = K = V = torch.randn(batch, seq_len, d_model)

    for num_heads in [1, 2, 4, 8]:
        output = multihead_attention(Q, K, V, num_heads)
        assert_shape(output, (batch, seq_len, d_model), f"num_heads={num_heads}")


def test_causal_mask():
    """Causal mask should prevent attending to future positions."""
    seq_len, d_model = 4, 16
    num_heads = 4

    Q = K = V = torch.randn(1, seq_len, d_model)

    # Upper triangular mask
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    output_masked = multihead_attention(Q, K, V, num_heads, mask=mask)
    output_unmasked = multihead_attention(Q, K, V, num_heads, mask=None)

    # Masked and unmasked should be different
    assert not torch.allclose(output_masked, output_unmasked), \
        "causal mask should change attention output"


def test_deterministic():
    """Same inputs should produce same outputs."""
    batch, seq_len, d_model = 2, 5, 32
    num_heads = 4

    torch.manual_seed(42)
    Q = K = V = torch.randn(batch, seq_len, d_model)

    output1 = multihead_attention(Q, K, V, num_heads)
    output2 = multihead_attention(Q, K, V, num_heads)

    assert_close(output1, output2, name="deterministic output")


def test_different_batch_sizes():
    """Should handle different batch sizes."""
    seq_len, d_model, num_heads = 5, 32, 4

    Q = K = V = torch.randn(1, seq_len, d_model)

    for batch in [1, 2, 8]:
        Q_batch = Q.expand(batch, -1, -1)
        K_batch = K.expand(batch, -1, -1)
        V_batch = V.expand(batch, -1, -1)

        output = multihead_attention(Q_batch, K_batch, V_batch, num_heads)
        assert_shape(output, (batch, seq_len, d_model), f"batch={batch}")


def test_no_nans_or_infs():
    """Output should not contain NaNs or infinities."""
    batch, seq_len, d_model = 2, 5, 64
    num_heads = 8

    Q = K = V = torch.randn(batch, seq_len, d_model)

    output = multihead_attention(Q, K, V, num_heads)

    assert not torch.isnan(output).any(), "output should not contain NaN"
    assert not torch.isinf(output).any(), "output should not contain inf"
