"""Batch normalization kata - reference solution."""

import torch
from jaxtyping import Float


def batch_norm_1d(
    x: Float[torch.Tensor, "batch features"],
    gamma: Float[torch.Tensor, "features"],
    beta: Float[torch.Tensor, "features"],
    eps: float = 1e-5,
) -> Float[torch.Tensor, "batch features"]:
    """Apply batch normalization to 2D input."""
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta


def batch_norm_2d(
    x: Float[torch.Tensor, "batch channels height width"],
    gamma: Float[torch.Tensor, "channels"],
    beta: Float[torch.Tensor, "channels"],
    eps: float = 1e-5,
) -> Float[torch.Tensor, "batch channels height width"]:
    """Apply batch normalization to 4D input (images)."""
    # Compute mean and var across (batch, height, width) for each channel
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)

    # Reshape gamma and beta for broadcasting
    gamma = gamma.view(1, -1, 1, 1)
    beta = beta.view(1, -1, 1, 1)

    return gamma * x_norm + beta
