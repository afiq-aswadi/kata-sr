"""Compute query-key similarity for attention mechanism."""

import torch
from jaxtyping import Float


def compute_qk_similarity(
    Q: Float[torch.Tensor, "batch seq_q d_model"],
    K: Float[torch.Tensor, "batch seq_k d_model"],
) -> Float[torch.Tensor, "batch seq_q seq_k"]:
    """Compute scaled dot-product similarity between queries and keys.

    Returns attention scores (before softmax).

    Args:
        Q: Query tensor
        K: Key tensor

    Returns:
        Attention scores of shape (batch, seq_q, seq_k)
    """
    # TODO: Implement Q @ K^T / sqrt(d_model)
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
