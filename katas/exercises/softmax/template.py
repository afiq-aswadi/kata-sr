"""Softmax implementation kata."""

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
    # TODO: implement numerically stable softmax
    # BLANK_START
    pass
    # BLANK_END
