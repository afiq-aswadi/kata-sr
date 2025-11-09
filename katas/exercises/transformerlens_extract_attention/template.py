"""Extract attention output activations from TransformerLens cache."""

import torch
from transformer_lens import HookedTransformer


def extract_attention_output(
    model: HookedTransformer, prompt: str, layer: int
) -> torch.Tensor:
    """Extract attention output (pre-projection) from a specific layer.

    Args:
        model: HookedTransformer model
        prompt: text prompt
        layer: layer number

    Returns:
        attention output tensor of shape (batch, seq_len, n_heads, d_head)
    """
    # BLANK_START
    raise NotImplementedError(
        "Run model with cache and extract attention output. "
        "Hint: cache key is f'blocks.{layer}.attn.hook_z'"
    )
    # BLANK_END
