"""Apply attention weights to value vectors."""

import torch
from jaxtyping import Float


def apply_attention_to_values(
    weights: Float[torch.Tensor, "batch seq_q seq_k"],
    V: Float[torch.Tensor, "batch seq_k d_model"],
) -> Float[torch.Tensor, "batch seq_q d_model"]:
    """Weight and sum value vectors using attention weights.

    Args:
        weights: Attention weights (probabilities)
        V: Value tensor

    Returns:
        Weighted sum of values of shape (batch, seq_q, d_model)
    """
    # TODO: Implement weights @ V
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
