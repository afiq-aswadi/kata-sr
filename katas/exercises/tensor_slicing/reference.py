"""Tensor slicing and indexing kata - reference solution."""

import torch
from jaxtyping import Bool, Float, Int


def gather_rows(
    x: Float[torch.Tensor, "batch features"], indices: Int[torch.Tensor, "batch"]
) -> Float[torch.Tensor, "batch features"]:
    """Select rows from a 2D tensor using indices."""
    return x[indices]


def scatter_add(
    x: Float[torch.Tensor, "n features"],
    indices: Int[torch.Tensor, "n"],
    output_size: int,
) -> Float[torch.Tensor, "output_size features"]:
    """Scatter and sum values into larger tensor."""
    features = x.shape[1]
    result = torch.zeros(output_size, features, dtype=x.dtype, device=x.device)
    result.scatter_add_(0, indices.unsqueeze(1).expand_as(x), x)
    return result


def masked_select_2d(
    x: Float[torch.Tensor, "h w"], mask: Bool[torch.Tensor, "h w"]
) -> Float[torch.Tensor, "selected"]:
    """Select elements where mask is True."""
    return x[mask]


def top_k_indices(
    x: Float[torch.Tensor, "batch features"], k: int
) -> Int[torch.Tensor, "batch k"]:
    """Get indices of top-k elements along last dimension."""
    return torch.topk(x, k, dim=-1).indices
