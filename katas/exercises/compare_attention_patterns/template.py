"""Compare attention patterns kata."""

import torch
from transformer_lens import HookedTransformer


def compare_attention_patterns(
    model: HookedTransformer,
    text1: str,
    text2: str,
    layer: int,
    head: int,
) -> dict[str, torch.Tensor]:
    """Compare attention patterns for two different prompts.

    Extract patterns for a specific head from both texts and compute
    their entropies for comparison.

    Args:
        model: HookedTransformer model
        text1: First input text
        text2: Second input text
        layer: Layer number to analyze
        head: Head number to analyze (0-indexed)

    Returns:
        Dictionary with keys:
            'pattern1': Attention pattern for text1, shape (seq1, seq1)
            'pattern2': Attention pattern for text2, shape (seq2, seq2)
            'entropy1': Entropy for text1, shape (seq1,)
            'entropy2': Entropy for text2, shape (seq2,)

    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> result = compare_attention_patterns(
        ...     model, "The cat", "The dog", layer=3, head=5
        ... )
        >>> result.keys()
        dict_keys(['pattern1', 'pattern2', 'entropy1', 'entropy2'])
    """
    # BLANK_START
    raise NotImplementedError("Extract patterns for both texts, compute entropies, return dict with specific head")
    # BLANK_END
