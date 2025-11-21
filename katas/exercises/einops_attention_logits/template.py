"""Scaled dot-product attention logits with einops."""

import math

import torch
from einops import einsum
from jaxtyping import Float


def scaled_attention_logits(
    query: Float[torch.Tensor, "batch heads q_len dim"],
    key: Float[torch.Tensor, "batch heads k_len dim"],
) -> Float[torch.Tensor, "batch heads q_len k_len"]:
    """Compute attention logits with sqrt(dim) scaling."""
    assert query.shape[0] == key.shape[0]
    assert query.shape[1] == key.shape[1]
    assert query.shape[3] == key.shape[3]

    # BLANK_START
    ...
    # BLANK_END
