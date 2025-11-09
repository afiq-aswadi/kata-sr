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

    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> patterns = extract_attention_pattern(model, "Hello world", layer=0)
        >>> patterns.shape
        torch.Size([1, 12, 3, 3])
    """
    # BLANK_START
    raise NotImplementedError(
        "Tokenize text, run with cache, extract cache[f'blocks.{layer}.attn.hook_pattern']"
    )
    # BLANK_END
