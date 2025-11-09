"""Extract attention patterns kata - reference solution."""

import torch
from transformer_lens import HookedTransformer


def extract_attention_patterns(
    model: HookedTransformer, text: str, layer: int
) -> torch.Tensor:
    """Extract attention patterns from a specific layer."""
    logits, cache = model.run_with_cache(text)
    return cache[f"blocks.{layer}.attn.hook_pattern"]
