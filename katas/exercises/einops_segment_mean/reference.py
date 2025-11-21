"""Reference for segment_mean."""

import torch
from einops import reduce
from jaxtyping import Float


def segment_mean(
    tokens: Float[torch.Tensor, "batch seq dim"],
    window_size: int,
) -> Float[torch.Tensor, "batch segments dim"]:
    assert window_size > 0
    batch, seq, _ = tokens.shape
    assert seq % window_size == 0

    return reduce(
        tokens,
        "b (segments window) d -> b segments d",
        "mean",
        window=window_size,
    )
