"""Tests for segment_mean."""

import pytest
import torch

try:
    from user_kata import segment_mean
except ImportError:
    from .reference import segment_mean


def test_basic_segment_mean():
    tokens = torch.tensor(
        [
            [
                [1.0, 3.0],
                [3.0, 5.0],
                [5.0, 7.0],
                [7.0, 9.0],
            ]
        ]
    )
    result = segment_mean(tokens, window_size=2)
    expected = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])
    assert torch.allclose(result, expected)


def test_multiple_batches_and_windows():
    tokens = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    result = segment_mean(tokens, window_size=3)
    assert result.shape == (2, 2, 3)

    # compute expected with manual reshape
    expected = tokens.reshape(2, 2, 3, 3).mean(dim=2)
    assert torch.allclose(result, expected)


def test_invalid_window_raises():
    tokens = torch.zeros(1, 5, 2)
    with pytest.raises(AssertionError):
        _ = segment_mean(tokens, window_size=4)
