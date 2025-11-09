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
    """
    batch_size, n_heads, seq_len, _ = patterns.shape

    # Extract attention to previous token for positions 1 onwards
    # For position i (i >= 1), get patterns[:, :, i, i-1]
    prev_token_attn = torch.zeros(batch_size, n_heads, seq_len - 1)
    for i in range(1, seq_len):
        prev_token_attn[:, :, i - 1] = patterns[:, :, i, i - 1]

    # Average across batch and query positions
    avg_prev_attn = prev_token_attn.mean(dim=(0, 2))  # (n_heads,)

    return avg_prev_attn > threshold
