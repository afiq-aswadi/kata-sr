"""Find previous token heads kata."""

import torch


def find_previous_token_heads(
    patterns: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Find attention heads that primarily attend to the previous token.

    A "previous token head" has high attention weight on the diagonal
    offset by 1 (each position attending strongly to position - 1).

    Args:
        patterns: Attention patterns of shape (batch, n_heads, seq, seq)
        threshold: Minimum average attention to previous token to qualify

    Returns:
        Boolean tensor of shape (n_heads,) indicating which heads
        attend to previous token

    Example:
        >>> # Create pattern where head 0 attends to previous token
        >>> patterns = torch.zeros(1, 2, 4, 4)
        >>> for i in range(1, 4):
        ...     patterns[0, 0, i, i-1] = 0.9  # Head 0: previous token
        ...     patterns[0, 1, i, i] = 0.9    # Head 1: self attention
        >>> prev_heads = find_previous_token_heads(patterns, threshold=0.5)
        >>> prev_heads  # tensor([True, False])
    """
    # BLANK_START
    raise NotImplementedError("Extract attention to position i-1 for each position i, average, compare to threshold")
    # BLANK_END
