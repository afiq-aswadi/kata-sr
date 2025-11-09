"""Use selective caching - reference solution."""

import torch
from transformer_lens import HookedTransformer


def cache_only_residual(model: HookedTransformer, prompt: str) -> dict:
    """Run model but only cache residual stream activations."""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(
        tokens, names_filter=lambda name: "resid" in name
    )
    return cache
