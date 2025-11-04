"""Tests for softmax kata."""

import torch
from jaxtyping import Float

from framework import assert_close, assert_shape
from user_kata import softmax


def test_output_shape():
    """Output should have same shape as input."""
    x = torch.randn(2, 3, 4)
    out = softmax(x, dim=-1)
    assert_shape(out, (2, 3, 4), "softmax output")


def test_sums_to_one():
    """Softmax output should sum to 1 along normalized dimension."""
    x = torch.randn(2, 3, 4)
    out = softmax(x, dim=-1)
    sums = out.sum(dim=-1)
    expected = torch.ones(2, 3)
    assert_close(sums, expected, name="softmax sums")


def test_all_positive():
    """All softmax values should be positive."""
    x = torch.randn(3, 5)
    out = softmax(x, dim=-1)
    assert (out > 0).all(), "softmax should output positive values"
    assert (out <= 1).all(), "softmax should output values <= 1"


def test_different_dimensions():
    """Test normalization along different dimensions."""
    x = torch.randn(2, 3, 4)

    # normalize along last dim
    out_last = softmax(x, dim=-1)
    assert_close(out_last.sum(dim=-1), torch.ones(2, 3), name="dim=-1 sums")

    # normalize along middle dim
    out_mid = softmax(x, dim=1)
    assert_close(out_mid.sum(dim=1), torch.ones(2, 4), name="dim=1 sums")

    # normalize along first dim
    out_first = softmax(x, dim=0)
    assert_close(out_first.sum(dim=0), torch.ones(3, 4), name="dim=0 sums")


def test_numerical_stability():
    """Softmax should handle large values without overflow."""
    x = torch.tensor([[1000.0, 1001.0, 1002.0]])
    out = softmax(x, dim=-1)

    # should not produce NaN or inf
    assert not torch.isnan(out).any(), "softmax produced NaN"
    assert not torch.isinf(out).any(), "softmax produced inf"

    # should still sum to 1
    assert_close(out.sum(dim=-1), torch.ones(1), name="large values sum")


def test_deterministic():
    """Same input should produce same output."""
    x = torch.randn(2, 5)
    out1 = softmax(x, dim=-1)
    out2 = softmax(x, dim=-1)
    assert_close(out1, out2, name="deterministic output")


def test_gradient_flow():
    """Gradients should flow through softmax."""
    x = torch.randn(2, 3, requires_grad=True)
    out = softmax(x, dim=-1)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "gradients should flow"
    assert not torch.isnan(x.grad).any(), "gradients should not be NaN"


def test_single_element():
    """Softmax of single element should be 1."""
    x = torch.tensor([[5.0]])
    out = softmax(x, dim=-1)
    expected = torch.tensor([[1.0]])
    assert_close(out, expected, name="single element")


def test_uniform_input():
    """Uniform input should produce uniform output."""
    x = torch.ones(2, 5)
    out = softmax(x, dim=-1)
    expected = torch.full((2, 5), 0.2)  # 1/5 for each element
    assert_close(out, expected, atol=1e-6, name="uniform input")


def test_max_dominates():
    """Element with maximum value should have highest probability."""
    x = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
    out = softmax(x, dim=-1)
    max_idx = out.argmax(dim=-1)
    assert max_idx.item() == 1, "maximum input should have maximum output"
