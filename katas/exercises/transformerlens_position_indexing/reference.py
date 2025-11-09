"""Extract position-specific activations - reference solution."""

import torch
from transformer_lens import HookedTransformer


def extract_position(
    model: HookedTransformer, prompt: str, layer: int, position: int
) -> torch.Tensor:
    """Extract residual stream at a specific token position."""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    residual = cache[f"blocks.{layer}.hook_resid_post"]
    return residual[0, position, :]  # [batch=0, position, :]
