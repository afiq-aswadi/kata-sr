"""Reference implementation for multi-head attention kata."""

import torch
import torch.nn as nn
from einops import einsum, rearrange
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

        # query, key, value projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # output projection
        self.w_o = nn.Linear(d_model, d_model)

    def forward(
        self, x: Float[torch.Tensor, "batch seq d_model"]
    ) -> Float[torch.Tensor, "batch seq d_model"]:
        """Forward pass.

        Args:
            x: input tensor

        Returns:
            output tensor with same shape as input
        """
        batch, seq_len, d_model = x.shape

        # project to Q, K, V and reshape for multi-head
        q = rearrange(self.w_q(x), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.w_k(x), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.w_v(x), "b s (h d) -> b h s d", h=self.num_heads)

        # compute scaled dot-product attention scores
        scores = einsum(q, k, "b h i d, b h j d -> b h i j") / (self.d_head**0.5)

        # apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)

        # apply attention weights to values
        out = einsum(attn_weights, v, "b h i j, b h j d -> b h i d")

        # concatenate heads and apply output projection
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.w_o(out)

        return out


def create_causal_mask(seq_len: int) -> Float[torch.Tensor, "seq seq"]:
    """Create upper triangular mask for causal attention.

    Args:
        seq_len: sequence length

    Returns:
        boolean mask tensor
    """
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
