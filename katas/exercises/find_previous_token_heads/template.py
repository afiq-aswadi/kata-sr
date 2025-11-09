"""Identify attention heads that attend to previous tokens."""

import torch


def find_previous_token_heads(
    patterns: torch.Tensor, threshold: float = 0.4
) -> torch.Tensor:
    """Identify attention heads that primarily attend to the previous token.

    Previous-token heads have high attention weight on the diagonal offset by 1.
    For each query position i (where i >= 1), check patterns[:, :, i, i-1].

    Args:
        patterns: attention patterns (batch, n_heads, query_pos, key_pos)
        threshold: minimum average attention to previous token to qualify

    Returns:
        boolean tensor (n_heads,) indicating which heads are previous-token heads

    Example:
        >>> patterns = torch.zeros(1, 3, 5, 5)
        >>> # Head 0 attends to previous token
        >>> for i in range(1, 5):
        ...     patterns[0, 0, i, i-1] = 0.9
        >>> result = find_previous_token_heads(patterns, threshold=0.5)
        >>> result[0]
        tensor(True)
    """
    # BLANK_START
    raise NotImplementedError(
        "Extract attention to previous token for each position, average across batch and positions"
    )
    # BLANK_END
