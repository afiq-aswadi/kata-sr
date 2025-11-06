"""Batch normalization kata."""

import torch
from jaxtyping import Float


def batch_norm_1d(
    x: Float[torch.Tensor, "batch features"],
    gamma: Float[torch.Tensor, "features"],
    beta: Float[torch.Tensor, "features"],
    eps: float = 1e-5,
) -> Float[torch.Tensor, "batch features"]:
    """Apply batch normalization to 2D input.

    Args:
        x: input tensor (batch, features)
        gamma: scale parameter
        beta: shift parameter
        eps: small constant for numerical stability

    Returns:
        normalized tensor
    """
    # TODO: normalize across batch dimension, then scale and shift
    # BLANK_START
    pass
    # BLANK_END


def batch_norm_2d(
    x: Float[torch.Tensor, "batch channels height width"],
    gamma: Float[torch.Tensor, "channels"],
    beta: Float[torch.Tensor, "channels"],
    eps: float = 1e-5,
) -> Float[torch.Tensor, "batch channels height width"]:
    """Apply batch normalization to 4D input (images).

    Args:
        x: input tensor (batch, channels, height, width)
        gamma: scale parameter per channel
        beta: shift parameter per channel
        eps: small constant for numerical stability

    Returns:
        normalized tensor
    """
    # TODO: normalize across (batch, height, width) for each channel
    # BLANK_START
    pass
    # BLANK_END
