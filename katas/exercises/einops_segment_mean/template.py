"""Segment-wise mean pooling with einops.reduce."""

import torch
from einops import reduce
from jaxtyping import Float


def segment_mean(
    tokens: Float[torch.Tensor, "batch seq dim"],
    window_size: int,
) -> Float[torch.Tensor, "batch segments dim"]:
    """Compute mean over non-overlapping windows along sequence."""
    assert window_size > 0
    batch, seq, _ = tokens.shape
    assert seq % window_size == 0

    # BLANK_START
    ...
    # BLANK_END
