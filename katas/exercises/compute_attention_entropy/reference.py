"""Compute attention entropy kata - reference solution."""

import torch


def compute_attention_entropy(patterns: torch.Tensor) -> torch.Tensor:
    """Compute entropy of attention patterns for each query position."""
    # Add small epsilon to prevent log(0)
    epsilon = 1e-10
    # Entropy = -sum(p * log(p))
    entropy = -(patterns * torch.log(patterns + epsilon)).sum(dim=-1)
    return entropy
