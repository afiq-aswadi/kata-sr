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

    Example:
        >>> patterns = torch.rand(1, 2, 5, 5)
        >>> patterns = patterns / patterns.sum(dim=-1, keepdim=True)
        >>> pos, weight = get_max_attention_position(patterns, query_pos=2, head=0)
        >>> 0 <= pos < 5
        True
        >>> 0.0 <= weight <= 1.0
        True
    """
    # BLANK_START
    raise NotImplementedError(
        "Extract attention weights for this query/head, find argmax and max value"
    )
    # BLANK_END
