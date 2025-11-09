"""Get max attention positions kata - reference solution."""

import torch


def get_max_attention_positions(
    patterns: torch.Tensor, top_k: int = 3
) -> torch.Tensor:
    """Find the top-k key positions each query attends to most strongly."""
    # torch.topk returns (values, indices)
    _, indices = torch.topk(patterns, k=top_k, dim=-1)
    return indices
