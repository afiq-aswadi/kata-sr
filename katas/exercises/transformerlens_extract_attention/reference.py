"""Extract attention output activations - reference solution."""

import torch
from transformer_lens import HookedTransformer


def extract_attention_output(
    model: HookedTransformer, prompt: str, layer: int
) -> torch.Tensor:
    """Extract attention output (pre-projection) from a specific layer."""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    return cache[f"blocks.{layer}.attn.hook_z"]
