"""Compare attention patterns between different prompts."""

import torch
from transformer_lens import HookedTransformer


def compare_attention_patterns(
    model: HookedTransformer,
    text1: str,
    text2: str,
    layer: int,
    head: int,
) -> float:
    """Compare attention patterns between two prompts using cosine similarity.

    Useful for understanding how attention changes with different inputs.

    Args:
        model: HookedTransformer model
        text1: first input text
        text2: second input text
        layer: layer number to compare
        head: head index to compare

    Returns:
        cosine similarity between patterns (scalar float)
        1.0 = identical patterns, 0.0 = orthogonal, -1.0 = opposite

    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> sim = compare_attention_patterns(model, "Hello", "Hello", 0, 0)
        >>> abs(sim - 1.0) < 0.01  # Same text should be nearly identical
        True
    """
    # BLANK_START
    raise NotImplementedError(
        "Extract patterns for both texts, truncate to same length, use cosine_similarity"
    )
    # BLANK_END
