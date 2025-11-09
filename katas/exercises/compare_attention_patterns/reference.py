"""Compare attention patterns kata - reference solution."""

import torch
from transformer_lens import HookedTransformer


def compare_attention_patterns(
    model: HookedTransformer,
    text1: str,
    text2: str,
    layer: int,
    head: int,
) -> dict[str, torch.Tensor]:
    """Compare attention patterns for two different prompts."""
    # Extract patterns for both texts
    from extract_attention_patterns.reference import extract_attention_patterns
    from compute_attention_entropy.reference import compute_attention_entropy

    patterns1 = extract_attention_patterns(model, text1, layer)
    patterns2 = extract_attention_patterns(model, text2, layer)

    # Compute entropies
    entropy1 = compute_attention_entropy(patterns1)
    entropy2 = compute_attention_entropy(patterns2)

    # Extract specific head (remove batch dimension, select head)
    pattern1 = patterns1[0, head, :, :]  # (seq1, seq1)
    pattern2 = patterns2[0, head, :, :]  # (seq2, seq2)
    entropy1_head = entropy1[0, head, :]  # (seq1,)
    entropy2_head = entropy2[0, head, :]  # (seq2,)

    return {
        'pattern1': pattern1,
        'pattern2': pattern2,
        'entropy1': entropy1_head,
        'entropy2': entropy2_head,
    }
