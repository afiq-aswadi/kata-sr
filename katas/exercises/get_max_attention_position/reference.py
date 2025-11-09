"""Find position receiving maximum attention from a query."""

import torch


def get_max_attention_position(
    patterns: torch.Tensor, query_pos: int, head: int, batch_idx: int = 0
) -> tuple[int, float]:
    """Find which position receives maximum attention from a query position.

    Args:
        patterns: attention patterns (batch, n_heads, query_pos, key_pos)
        query_pos: query position to analyze (0-indexed)
        head: head index to analyze (0-indexed)
        batch_idx: batch index to analyze (default: 0)

    Returns:
        tuple of (position receiving max attention, attention weight)
    """
    attn_weights = patterns[batch_idx, head, query_pos, :]
    max_pos = attn_weights.argmax().item()
    max_weight = attn_weights[max_pos].item()
    return max_pos, max_weight
