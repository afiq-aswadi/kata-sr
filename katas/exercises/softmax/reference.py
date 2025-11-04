"""Reference implementation for softmax kata."""

import torch
from jaxtyping import Float


def softmax(
    x: Float[torch.Tensor, "..."], dim: int = -1
) -> Float[torch.Tensor, "..."]:
    """Apply softmax normalization along specified dimension.

    Args:
        x: input tensor of any shape
        dim: dimension to normalize over

    Returns:
        tensor with same shape as input, normalized along dim
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
