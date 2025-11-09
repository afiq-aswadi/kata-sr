"""Compute attention entropy kata."""

import torch


def compute_attention_entropy(patterns: torch.Tensor) -> torch.Tensor:
    """Compute entropy of attention patterns for each query position.

    Entropy measures how focused or diffuse the attention is:
    - Low entropy: attention focused on few tokens
    - High entropy: attention spread across many tokens

    Args:
        patterns: Attention patterns of shape (batch, n_heads, seq_q, seq_k)
                  Should be normalized (sum to 1.0 across seq_k)

    Returns:
        Entropy tensor of shape (batch, n_heads, seq_q)
        Formula: -sum(p * log(p)) for each query position

    Example:
        >>> # Focused attention (low entropy)
        >>> focused = torch.tensor([[[[1.0, 0.0, 0.0]]]])
        >>> entropy = compute_attention_entropy(focused)
        >>> entropy  # ~0.0 (very focused)

        >>> # Uniform attention (high entropy)
        >>> uniform = torch.tensor([[[[0.33, 0.33, 0.34]]]])
        >>> entropy = compute_attention_entropy(uniform)
        >>> entropy  # ~1.1 (diffuse)
    """
    # BLANK_START
    raise NotImplementedError("Compute -sum(p * log(p + epsilon)) over key dimension")
    # BLANK_END
