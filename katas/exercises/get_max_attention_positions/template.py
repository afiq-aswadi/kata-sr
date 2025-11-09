"""Get max attention positions kata."""

import torch


def get_max_attention_positions(
    patterns: torch.Tensor, top_k: int = 3
) -> torch.Tensor:
    """Find the top-k key positions each query attends to most strongly.

    For each query position, identify which key positions receive
    the highest attention weights.

    Args:
        patterns: Attention patterns of shape (batch, n_heads, seq_q, seq_k)
        top_k: Number of top positions to return (default: 3)

    Returns:
        Indices tensor of shape (batch, n_heads, seq_q, top_k)
        Contains the key position indices with highest attention weights

    Example:
        >>> patterns = torch.tensor([[[[0.1, 0.5, 0.3, 0.1]]]])  # (1,1,1,4)
        >>> top_2 = get_max_attention_positions(patterns, top_k=2)
        >>> top_2[0, 0, 0]  # For query 0
        tensor([1, 2])  # Position 1 (0.5) and position 2 (0.3) are top-2
    """
    # BLANK_START
    raise NotImplementedError("Use torch.topk on last dimension, return indices")
    # BLANK_END
