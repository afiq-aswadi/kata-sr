"""Reference solution for applying attention to values."""

import torch
from jaxtyping import Float


def apply_attention_to_values(
    weights: Float[torch.Tensor, "batch seq_q seq_k"],
    V: Float[torch.Tensor, "batch seq_k d_model"],
) -> Float[torch.Tensor, "batch seq_q d_model"]:
    """Weight and sum value vectors using attention weights."""
    return torch.matmul(weights, V)
