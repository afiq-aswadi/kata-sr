"""Reference for scaled_attention_logits."""

import math

import torch
from einops import einsum
from jaxtyping import Float


def scaled_attention_logits(
    query: Float[torch.Tensor, "batch heads q_len dim"],
    key: Float[torch.Tensor, "batch heads k_len dim"],
) -> Float[torch.Tensor, "batch heads q_len k_len"]:
    assert query.shape[0] == key.shape[0]
    assert query.shape[1] == key.shape[1]
    assert query.shape[3] == key.shape[3]

    dim = query.shape[-1]
    scale = 1.0 / math.sqrt(dim)
    return einsum(query, key, "b h q d, b h k d -> b h q k") * scale
