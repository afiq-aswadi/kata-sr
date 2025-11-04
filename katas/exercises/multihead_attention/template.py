"""Multi-head attention implementation kata."""

import torch
import torch.nn as nn
from jaxtyping import Float


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module.

    Args:
        d_model: dimension of input embeddings
        num_heads: number of attention heads
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # TODO: initialize query, key, value, and output projections
        # BLANK_START
        raise NotImplementedError
        # BLANK_END

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq d_model"],
        mask: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, "batch seq d_model"]:
        """Forward pass.

        Args:
            x: input tensor
            mask: optional attention mask where True entries are masked

        Returns:
            output tensor with same shape as input
        """
        # TODO: implement multi-head attention
        # BLANK_START
        raise NotImplementedError
        # BLANK_END


def create_causal_mask(seq_len: int) -> Float[torch.Tensor, "seq seq"]:
    """Create upper triangular mask for causal attention.

    Args:
        seq_len: sequence length

    Returns:
        boolean mask tensor
    """
    # TODO: create causal mask
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
