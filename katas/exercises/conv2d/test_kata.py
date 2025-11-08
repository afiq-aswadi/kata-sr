"""Tests for conv2d kata."""

import torch
import torch.nn.functional as F


def test_conv2d_basic():
    from template import conv2d

    x = torch.randn(2, 3, 8, 8)
    kernel = torch.randn(4, 3, 3, 3)
    result = conv2d(x, kernel, stride=1, padding=0)
    expected = F.conv2d(x, kernel, stride=1, padding=0)
    assert torch.allclose(result, expected, atol=1e-5)


def test_conv2d_with_padding():
    from template import conv2d

    x = torch.randn(1, 1, 5, 5)
    kernel = torch.randn(1, 1, 3, 3)
    result = conv2d(x, kernel, stride=1, padding=1)
    expected = F.conv2d(x, kernel, stride=1, padding=1)
    assert torch.allclose(result, expected, atol=1e-5)


def test_conv2d_with_stride():
    from template import conv2d

    x = torch.randn(1, 2, 10, 10)
    kernel = torch.randn(3, 2, 3, 3)
    result = conv2d(x, kernel, stride=2, padding=0)
    expected = F.conv2d(x, kernel, stride=2, padding=0)
    assert torch.allclose(result, expected, atol=1e-5)


def test_conv2d_einsum_basic():
    from template import conv2d_einsum

    x = torch.randn(2, 3, 6, 6)
    kernel = torch.randn(4, 3, 3, 3)
    result = conv2d_einsum(x, kernel)
    expected = F.conv2d(x, kernel)
    assert torch.allclose(result, expected, atol=1e-5)


def test_conv2d_output_shape():
    from template import conv2d

    x = torch.randn(1, 1, 10, 10)
    kernel = torch.randn(1, 1, 3, 3)
    result = conv2d(x, kernel, stride=1, padding=0)
    assert result.shape == (1, 1, 8, 8)
