"""Reference solution for attention weights."""

import torch
from jaxtyping import Bool, Float


def compute_attention_weights(
    scores: Float[torch.Tensor, "batch seq_q seq_k"],
    mask: Bool[torch.Tensor, "seq_q seq_k"] | None = None,
) -> Float[torch.Tensor, "batch seq_q seq_k"]:
    """Apply softmax to get attention weights, with optional masking."""
    if mask is not None:
        # Expand mask to match batch dimension
        mask_expanded = mask.unsqueeze(0).expand(scores.shape[0], -1, -1)
        scores = scores.masked_fill(mask_expanded, float('-inf'))

    weights = torch.softmax(scores, dim=-1)
    return weights
