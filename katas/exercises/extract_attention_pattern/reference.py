"""Extract attention patterns from TransformerLens cache."""

import torch
from transformer_lens import HookedTransformer


def extract_attention_pattern(
    model: HookedTransformer, text: str, layer: int
) -> torch.Tensor:
    """Extract attention patterns from a specific layer.

    Args:
        model: HookedTransformer model
        text: input text to process
        layer: layer number to extract from (0-indexed)

    Returns:
        attention patterns of shape (batch, n_heads, query_pos, key_pos)
        These are post-softmax probabilities that sum to 1.0 across key dimension.
    """
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)
    patterns = cache[f"blocks.{layer}.attn.hook_pattern"]
    return patterns
