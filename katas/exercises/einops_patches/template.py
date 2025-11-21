"""Convert images into flattened patches with einops."""

import torch
from einops import rearrange
from jaxtyping import Float


def image_to_patches(
    images: Float[torch.Tensor, "batch channels height width"],
    patch_size: int,
) -> Float[torch.Tensor, "batch num_patches patch_dim"]:
    """Split images into non-overlapping flattened patches.

    Args:
        images: input tensor shaped (batch, channels, height, width).
        patch_size: edge length of each square patch. Both height and width
            must be divisible by patch_size.

    Returns:
        Tensor of shape (batch, num_patches, channels * patch_size * patch_size).
    """
    assert patch_size > 0
    batch, channels, height, width = images.shape
    assert height % patch_size == 0
    assert width % patch_size == 0

    # BLANK_START
    ...
    # BLANK_END
