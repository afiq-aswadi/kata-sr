"""Reference solution for image_to_patches."""

import torch
from einops import rearrange
from jaxtyping import Float


def image_to_patches(
    images: Float[torch.Tensor, "batch channels height width"],
    patch_size: int,
) -> Float[torch.Tensor, "batch num_patches patch_dim"]:
    assert patch_size > 0
    batch, channels, height, width = images.shape
    assert height % patch_size == 0
    assert width % patch_size == 0

    return rearrange(
        images,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
    )
