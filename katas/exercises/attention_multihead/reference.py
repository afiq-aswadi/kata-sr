"""Reference solution for multi-head attention."""

import torch
from jaxtyping import Bool, Float


def multihead_attention(
    Q: Float[torch.Tensor, "batch seq_q d_model"],
    K: Float[torch.Tensor, "batch seq_k d_model"],
    V: Float[torch.Tensor, "batch seq_k d_model"],
    num_heads: int,
    mask: Bool[torch.Tensor, "seq_q seq_k"] | None = None,
) -> Float[torch.Tensor, "batch seq_q d_model"]:
    """Full multi-head attention using previous building blocks."""
    batch, seq_q, d_model = Q.shape
    seq_k = K.shape[1]
    d_head = d_model // num_heads

    # Reshape to (batch, num_heads, seq, d_head)
    Q_heads = Q.view(batch, seq_q, num_heads, d_head).transpose(1, 2)
    K_heads = K.view(batch, seq_k, num_heads, d_head).transpose(1, 2)
    V_heads = V.view(batch, seq_k, num_heads, d_head).transpose(1, 2)

    # Compute attention scores
    scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / (d_head**0.5)

    # Apply mask if provided
    if mask is not None:
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_q, seq_k)
        scores = scores.masked_fill(mask_expanded, float('-inf'))

    # Compute attention weights
    weights = torch.softmax(scores, dim=-1)

    # Apply attention to values
    attended = torch.matmul(weights, V_heads)

    # Concatenate heads: (batch, num_heads, seq_q, d_head) -> (batch, seq_q, d_model)
    output = attended.transpose(1, 2).contiguous().view(batch, seq_q, d_model)

    return output
