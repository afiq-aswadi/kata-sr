"""Compute attention entropy to measure focus vs diffusion."""

import torch


def compute_attention_entropy(patterns: torch.Tensor) -> torch.Tensor:
    """Compute entropy of attention patterns for each query position.

    Entropy measures how "focused" vs "diffuse" attention is:
    - High entropy = attention spread across many tokens
    - Low entropy = attention focused on few tokens

    Args:
        patterns: attention patterns (batch, n_heads, query_pos, key_pos)
                 Post-softmax probabilities that sum to 1.0 across key_pos

    Returns:
        entropy for each query position (batch, n_heads, query_pos)
        Formula: -sum(p * log(p)) across key dimension
    """
    # Add small epsilon to avoid log(0)
    entropy = -(patterns * torch.log(patterns + 1e-10)).sum(dim=-1)
    return entropy
