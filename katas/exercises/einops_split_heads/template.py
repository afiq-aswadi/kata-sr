"""Split merged attention heads with einops."""

import torch
from einops import rearrange
from jaxtyping import Float


def split_heads(
    hidden: Float[torch.Tensor, "batch seq hidden_dim"],
    num_heads: int,
) -> Float[torch.Tensor, "batch heads seq head_dim"]:
    """Reshape merged heads into (heads, head_dim)."""
    assert num_heads > 0
    _, _, hidden_dim = hidden.shape
    assert hidden_dim % num_heads == 0

    # BLANK_START
    ...
    # BLANK_END
