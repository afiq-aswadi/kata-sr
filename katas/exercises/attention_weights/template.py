"""Apply softmax to attention scores with optional masking."""

import torch
from jaxtyping import Bool, Float


def compute_attention_weights(
    scores: Float[torch.Tensor, "batch seq_q seq_k"],
    mask: Bool[torch.Tensor, "seq_q seq_k"] | None = None,
) -> Float[torch.Tensor, "batch seq_q seq_k"]:
    """Apply softmax to get attention weights, with optional masking.

    Args:
        scores: Attention scores (before softmax)
        mask: Optional boolean mask where True entries are masked out

    Returns:
        Attention weights (probabilities) of shape (batch, seq_q, seq_k)
    """
    # TODO: Apply mask (set masked positions to -inf), then softmax
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
