"""Tests for image_to_patches."""

import pytest
import torch
from einops import rearrange

try:
    from user_kata import image_to_patches
except ImportError:
    from .reference import image_to_patches


def test_single_image_patches_match_expected():
    images = torch.arange(16.0).reshape(1, 1, 4, 4)
    result = image_to_patches(images, patch_size=2)
    expected = torch.tensor(
        [
            [
                [0.0, 1.0, 4.0, 5.0],
                [2.0, 3.0, 6.0, 7.0],
                [8.0, 9.0, 12.0, 13.0],
                [10.0, 11.0, 14.0, 15.0],
            ]
        ]
    )
    assert torch.equal(result, expected)


def test_batch_and_channels_shape_and_values():
    images = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    result = image_to_patches(images, patch_size=2)

    assert result.shape == (2, 4, 12)

    expected = rearrange(images, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    assert torch.equal(result, expected)


def test_invalid_patch_size_raises():
    images = torch.zeros(1, 1, 3, 3)
    with pytest.raises(AssertionError):
        _ = image_to_patches(images, patch_size=2)
