"""Compute attribution scores across all layers and positions."""

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer


def compute_attribution_scores(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    metric_fn: callable,
) -> dict[str, Float[torch.Tensor, "..."]]:
    """Score all layers/positions for attribution.

    Systematically patch clean activations into corrupted run across all
    layers and positions, measuring impact using a metric function.

    Args:
        model: HookedTransformer model
        clean_text: clean input
        corrupted_text: corrupted input
        metric_fn: function that takes logits and returns a scalar score

    Returns:
        Dictionary mapping layer names to attribution score tensors
        Each tensor has shape (seq_len,) with scores for each position
    """
    # TODO:
    # 1. Get baseline metric scores (clean and corrupted)
    # 2. For each layer, patch each position and measure metric change
    # 3. Compute attribution score as (patched - corrupted) / (clean - corrupted)
    # 4. Return dict of layer -> scores
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
