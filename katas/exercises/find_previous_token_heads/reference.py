"""Find previous token heads kata - reference solution."""

import torch


def find_previous_token_heads(
    patterns: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Find attention heads that primarily attend to the previous token."""
    batch, n_heads, seq_len, _ = patterns.shape

    # Handle edge case: single token
    if seq_len <= 1:
        return torch.zeros(n_heads, dtype=torch.bool)

    # For positions 1 onwards, get attention to previous token (diagonal offset by 1)
    prev_token_attn = []
    for i in range(1, seq_len):
        prev_token_attn.append(patterns[:, :, i, i-1])

    # Stack and average across positions and batch
    prev_token_attn = torch.stack(prev_token_attn, dim=-1)  # (batch, n_heads, seq-1)
    avg_prev_token_attn = prev_token_attn.mean(dim=(0, 2))  # (n_heads,)

    # Check which heads exceed threshold
    return avg_prev_token_attn > threshold
