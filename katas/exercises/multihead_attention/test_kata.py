"""Tests for multi-head attention kata."""

import torch

from framework import assert_close, assert_shape
from user_kata import MultiHeadAttention, create_causal_mask


def test_output_shape():
    """Output should have same shape as input."""
    batch, seq_len, d_model = 2, 10, 64
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model)
    out = mha(x)

    assert_shape(out, (batch, seq_len, d_model), "attention output")


def test_different_sequence_lengths():
    """Attention should handle different sequence lengths."""
    d_model, num_heads = 64, 8
    mha = MultiHeadAttention(d_model, num_heads)

    for seq_len in [1, 5, 10, 20]:
        x = torch.randn(2, seq_len, d_model)
        out = mha(x)
        assert_shape(out, (2, seq_len, d_model), f"seq_len={seq_len}")


def test_different_batch_sizes():
    """Attention should handle different batch sizes."""
    seq_len, d_model, num_heads = 10, 64, 8
    mha = MultiHeadAttention(d_model, num_heads)

    for batch in [1, 4, 16]:
        x = torch.randn(batch, seq_len, d_model)
        out = mha(x)
        assert_shape(out, (batch, seq_len, d_model), f"batch={batch}")


def test_gradient_flow():
    """Gradients should flow through attention."""
    batch, seq_len, d_model = 2, 5, 16
    num_heads = 4

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)

    out = mha(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "gradients should flow to input"
    assert not torch.isnan(x.grad).any(), "gradients should not be NaN"


def test_deterministic():
    """Same input should produce same output."""
    batch, seq_len, d_model = 2, 5, 16
    num_heads = 4

    torch.manual_seed(42)
    mha = MultiHeadAttention(d_model, num_heads)

    x = torch.randn(batch, seq_len, d_model)
    out1 = mha(x)
    out2 = mha(x)

    assert_close(out1, out2, name="deterministic output")


def test_different_num_heads():
    """Attention should work with different number of heads."""
    batch, seq_len, d_model = 2, 5, 64

    for num_heads in [1, 2, 4, 8]:
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch, seq_len, d_model)
        out = mha(x)
        assert_shape(out, (batch, seq_len, d_model), f"num_heads={num_heads}")


def test_single_head():
    """Attention should work with single head (degenerates to regular attention)."""
    batch, seq_len, d_model = 2, 5, 16
    mha = MultiHeadAttention(d_model, num_heads=1)

    x = torch.randn(batch, seq_len, d_model)
    out = mha(x)
    assert_shape(out, (batch, seq_len, d_model), "single head")


def test_no_nans_or_infs():
    """Output should not contain NaNs or infinities."""
    batch, seq_len, d_model = 2, 10, 64
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model)
    out = mha(x)

    assert not torch.isnan(out).any(), "output should not contain NaN"
    assert not torch.isinf(out).any(), "output should not contain inf"


def test_causal_mask_shape():
    """Causal mask should have correct shape."""
    seq_len = 5
    mask = create_causal_mask(seq_len)
    assert_shape(mask, (seq_len, seq_len), "causal mask")


def test_causal_mask_structure():
    """Causal mask should be upper triangular."""
    mask = create_causal_mask(4)
    expected = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    assert torch.equal(mask, expected), "causal mask should be upper triangular"


def test_causal_mask_single_element():
    """Causal mask for single element should be all False."""
    mask = create_causal_mask(1)
    expected = torch.tensor([[False]])
    assert torch.equal(mask, expected), "single element mask"
