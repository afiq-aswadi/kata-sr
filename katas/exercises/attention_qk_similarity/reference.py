"""Reference solution for QK similarity."""

import torch
from jaxtyping import Float


def compute_qk_similarity(
    Q: Float[torch.Tensor, "batch seq_q d_model"],
    K: Float[torch.Tensor, "batch seq_k d_model"],
) -> Float[torch.Tensor, "batch seq_q seq_k"]:
    """Compute scaled dot-product similarity between queries and keys."""
    d_model = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_model**0.5)
    return scores
