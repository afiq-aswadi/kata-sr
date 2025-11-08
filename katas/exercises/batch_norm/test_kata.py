"""Tests for batch normalization kata."""

import torch


def test_batch_norm_1d_basic():
    from template import batch_norm_1d

    x = torch.randn(32, 10)
    gamma = torch.ones(10)
    beta = torch.zeros(10)

    result = batch_norm_1d(x, gamma, beta)

    # Check normalization: mean should be ~0, std should be ~1
    assert torch.allclose(result.mean(dim=0), torch.zeros(10), atol=1e-5)
    assert torch.allclose(result.std(dim=0, unbiased=False), torch.ones(10), atol=1e-5)


def test_batch_norm_1d_with_affine():
    from template import batch_norm_1d

    x = torch.randn(16, 5)
    gamma = torch.randn(5) + 1.0
    beta = torch.randn(5)

    result = batch_norm_1d(x, gamma, beta)
    assert result.shape == x.shape


def test_batch_norm_2d_basic():
    from template import batch_norm_2d

    x = torch.randn(8, 3, 16, 16)
    gamma = torch.ones(3)
    beta = torch.zeros(3)

    result = batch_norm_2d(x, gamma, beta)

    # Check shape
    assert result.shape == x.shape

    # Check normalization per channel
    for c in range(3):
        channel_data = result[:, c, :, :]
        assert torch.allclose(
            channel_data.mean(), torch.tensor(0.0), atol=1e-5
        )
        assert torch.allclose(
            channel_data.std(unbiased=False), torch.tensor(1.0), atol=1e-5
        )


def test_batch_norm_2d_with_affine():
    from template import batch_norm_2d

    x = torch.randn(4, 6, 8, 8)
    gamma = torch.randn(6) + 1.0
    beta = torch.randn(6)

    result = batch_norm_2d(x, gamma, beta)
    assert result.shape == x.shape


def test_batch_norm_1d_single_batch():
    from template import batch_norm_1d

    x = torch.tensor([[1.0, 2.0, 3.0]])
    gamma = torch.ones(3)
    beta = torch.zeros(3)

    result = batch_norm_1d(x, gamma, beta)
    # With single batch, all values should become 0 after normalization
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)
