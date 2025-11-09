"""Use selective caching to reduce memory usage in TransformerLens."""

import torch
from transformer_lens import HookedTransformer


def cache_only_residual(model: HookedTransformer, prompt: str) -> dict:
    """Run model but only cache residual stream activations.

    Args:
        model: HookedTransformer model
        prompt: text prompt

    Returns:
        cache containing only residual stream activations (not attention or MLP)
    """
    # BLANK_START
    raise NotImplementedError(
        "Use run_with_cache with names_filter parameter. "
        "Hint: names_filter=lambda name: 'resid' in name"
    )
    # BLANK_END
