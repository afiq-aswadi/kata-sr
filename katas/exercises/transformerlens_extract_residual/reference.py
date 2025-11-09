"""Extract residual stream activations - reference solution."""

import torch
from transformer_lens import HookedTransformer


def extract_residual_stream(
    model: HookedTransformer, prompt: str, layer: int
) -> torch.Tensor:
    """Extract residual stream activations from a specific layer."""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    return cache[f"blocks.{layer}.hook_resid_post"]
