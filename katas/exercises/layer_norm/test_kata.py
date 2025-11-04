"""Tests for layer normalization kata."""

import torch

from framework import assert_close, assert_shape
from user_kata import LayerNorm


def test_output_shape():
    """Output should have same shape as input."""
    ln = LayerNorm(normalized_shape=10)
    x = torch.randn(2, 5, 10)
    out = ln(x)
    assert_shape(out, (2, 5, 10), "layer_norm output")


def test_normalized_mean():
    """Normalized output should have mean close to 0 (before scale/shift)."""
    # use gamma=1, beta=0 to test raw normalization
    ln = LayerNorm(normalized_shape=10)
    ln.gamma.data.fill_(1.0)
    ln.beta.data.fill_(0.0)

    x = torch.randn(3, 4, 10)
    out = ln(x)

    # mean across feature dimension should be close to 0
    mean = out.mean(dim=-1)
    expected = torch.zeros(3, 4)
    assert_close(mean, expected, atol=1e-5, name="normalized mean")


def test_normalized_variance():
    """Normalized output should have variance close to 1 (before scale/shift)."""
    ln = LayerNorm(normalized_shape=10)
    ln.gamma.data.fill_(1.0)
    ln.beta.data.fill_(0.0)

    x = torch.randn(3, 4, 10)
    out = ln(x)

    # variance across feature dimension should be close to 1
    var = out.var(dim=-1, unbiased=False)
    expected = torch.ones(3, 4)
    assert_close(var, expected, rtol=1e-4, name="normalized variance")


def test_learnable_parameters():
    """Gamma and beta should be learnable parameters."""
    ln = LayerNorm(normalized_shape=10)

    params = dict(ln.named_parameters())
    assert "gamma" in params, "gamma should be a parameter"
    assert "beta" in params, "beta should be a parameter"

    assert params["gamma"].requires_grad, "gamma should be trainable"
    assert params["beta"].requires_grad, "beta should be trainable"

    assert params["gamma"].shape == (10,), "gamma shape should match normalized_shape"
    assert params["beta"].shape == (10,), "beta shape should match normalized_shape"


def test_scale_and_shift():
    """Gamma scales and beta shifts the normalized output."""
    ln = LayerNorm(normalized_shape=5)

    # set specific gamma and beta values
    ln.gamma.data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    ln.beta.data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    # use input that normalizes to known values
    x = torch.ones(1, 5)  # uniform input
    out = ln(x)

    # with uniform input, after normalization all values should be 0
    # then scaled by gamma and shifted by beta
    expected = ln.beta  # 0 * gamma + beta = beta
    assert_close(out[0], expected, atol=1e-5, name="scale and shift")


def test_gradient_flow():
    """Gradients should flow through normalization."""
    ln = LayerNorm(normalized_shape=10)
    x = torch.randn(2, 5, 10, requires_grad=True)

    out = ln(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "gradients should flow to input"
    assert ln.gamma.grad is not None, "gradients should flow to gamma"
    assert ln.beta.grad is not None, "gradients should flow to beta"


def test_different_input_shapes():
    """Layer norm should work with different batch dimensions."""
    ln = LayerNorm(normalized_shape=10)

    # 2D: (batch, features)
    x_2d = torch.randn(5, 10)
    out_2d = ln(x_2d)
    assert_shape(out_2d, (5, 10), "2D input")

    # 3D: (batch, seq, features)
    x_3d = torch.randn(2, 5, 10)
    out_3d = ln(x_3d)
    assert_shape(out_3d, (2, 5, 10), "3D input")

    # 4D: (batch, height, width, features)
    x_4d = torch.randn(2, 3, 4, 10)
    out_4d = ln(x_4d)
    assert_shape(out_4d, (2, 3, 4, 10), "4D input")


def test_numerical_stability():
    """Layer norm should handle extreme values."""
    ln = LayerNorm(normalized_shape=5)

    # large values
    x_large = torch.tensor([[1000.0, 1001.0, 1002.0, 1003.0, 1004.0]])
    out_large = ln(x_large)
    assert not torch.isnan(out_large).any(), "should handle large values"
    assert not torch.isinf(out_large).any(), "should not produce inf"

    # small values
    x_small = torch.tensor([[1e-6, 2e-6, 3e-6, 4e-6, 5e-6]])
    out_small = ln(x_small)
    assert not torch.isnan(out_small).any(), "should handle small values"


def test_deterministic():
    """Same input should produce same output."""
    torch.manual_seed(42)
    ln = LayerNorm(normalized_shape=10)

    x = torch.randn(2, 5, 10)
    out1 = ln(x)
    out2 = ln(x)

    assert_close(out1, out2, name="deterministic output")


def test_different_from_input():
    """Normalized output should generally differ from input."""
    ln = LayerNorm(normalized_shape=10)
    x = torch.randn(2, 5, 10)
    out = ln(x)

    # output should not be identical to input (unless by extreme coincidence)
    assert not torch.allclose(out, x, rtol=0.1), "output should differ from input"
