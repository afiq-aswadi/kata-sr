"""Multi-head attention using building blocks."""

import torch
from jaxtyping import Bool, Float


def multihead_attention(
    Q: Float[torch.Tensor, "batch seq_q d_model"],
    K: Float[torch.Tensor, "batch seq_k d_model"],
    V: Float[torch.Tensor, "batch seq_k d_model"],
    num_heads: int,
    mask: Bool[torch.Tensor, "seq_q seq_k"] | None = None,
) -> Float[torch.Tensor, "batch seq_q d_model"]:
    """Full multi-head attention using previous building blocks.

    Split Q, K, V into multiple heads, compute attention for each head,
    then concatenate the results.

    Args:
        Q: Query tensor
        K: Key tensor
        V: Value tensor
        num_heads: Number of attention heads
        mask: Optional attention mask

    Returns:
        Multi-head attention output of shape (batch, seq_q, d_model)
    """
    # TODO: Split into heads, apply attention, concat
    # 1. Split Q, K, V into heads: reshape to (batch, num_heads, seq, d_head)
    # 2. Compute attention for each head using previous building blocks
    # 3. Concatenate heads and reshape back to (batch, seq_q, d_model)
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
