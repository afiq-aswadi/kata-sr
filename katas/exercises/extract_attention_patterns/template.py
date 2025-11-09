"""Extract attention patterns kata."""

import torch
from transformer_lens import HookedTransformer


def extract_attention_patterns(
    model: HookedTransformer, text: str, layer: int
) -> torch.Tensor:
    """Extract attention patterns from a specific layer.

    Args:
        model: HookedTransformer model
        text: input text to process
        layer: layer number to extract from (0-indexed)

    Returns:
        Attention patterns of shape (batch, n_heads, seq, seq)
        These are post-softmax weights (already normalized)

    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> patterns = extract_attention_patterns(model, "Hello world", layer=0)
        >>> patterns.shape
        torch.Size([1, 12, 3, 3])  # batch=1, heads=12, seq=3
    """
    # BLANK_START
    raise NotImplementedError("Use run_with_cache and access cache[f'blocks.{layer}.attn.hook_pattern']")
    # BLANK_END
