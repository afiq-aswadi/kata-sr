"""Tensor slicing and indexing kata."""

import torch
from jaxtyping import Float, Int, Bool


def gather_rows(
    x: Float[torch.Tensor, "batch features"], indices: Int[torch.Tensor, "batch"]
) -> Float[torch.Tensor, "batch features"]:
    """Select rows from a 2D tensor using indices.

    Args:
        x: input tensor of shape (batch, features)
        indices: indices to select, shape (batch,)

    Returns:
        selected rows, shape (batch, features)
    """
    # TODO: use torch.gather or advanced indexing
    # BLANK_START
    pass
    # BLANK_END


def scatter_add(
    x: Float[torch.Tensor, "n features"],
    indices: Int[torch.Tensor, "n"],
    output_size: int,
) -> Float[torch.Tensor, "output_size features"]:
    """Scatter and sum values into larger tensor.

    Args:
        x: values to scatter, shape (n, features)
        indices: destination indices, shape (n,)
        output_size: size of output dimension 0

    Returns:
        scattered tensor, shape (output_size, features)
    """
    # TODO: use torch.scatter_add or zeros + indexing
    # BLANK_START
    pass
    # BLANK_END


def masked_select_2d(
    x: Float[torch.Tensor, "h w"], mask: Bool[torch.Tensor, "h w"]
) -> Float[torch.Tensor, "selected"]:
    """Select elements where mask is True.

    Args:
        x: input tensor
        mask: boolean mask

    Returns:
        1D tensor of selected elements
    """
    # TODO: use torch.masked_select or boolean indexing
    # BLANK_START
    pass
    # BLANK_END


def top_k_indices(
    x: Float[torch.Tensor, "batch features"], k: int
) -> Int[torch.Tensor, "batch k"]:
    """Get indices of top-k elements along last dimension.

    Args:
        x: input tensor
        k: number of top elements

    Returns:
        indices of top-k elements per row
    """
    # TODO: use torch.topk
    # BLANK_START
    pass
    # BLANK_END
