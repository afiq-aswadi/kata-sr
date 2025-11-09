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
    """
    # Extract patterns for both texts
    tokens1 = model.to_tokens(text1)
    _, cache1 = model.run_with_cache(tokens1)
    patterns1 = cache1[f"blocks.{layer}.attn.hook_pattern"]

    tokens2 = model.to_tokens(text2)
    _, cache2 = model.run_with_cache(tokens2)
    patterns2 = cache2[f"blocks.{layer}.attn.hook_pattern"]

    # Extract patterns for specific head
    pattern1 = patterns1[0, head]  # (seq1, seq1)
    pattern2 = patterns2[0, head]  # (seq2, seq2)

    # Handle different sequence lengths by using minimum
    min_seq_len = min(pattern1.shape[0], pattern2.shape[0])
    pattern1 = pattern1[:min_seq_len, :min_seq_len]
    pattern2 = pattern2[:min_seq_len, :min_seq_len]

    # Flatten patterns for cosine similarity
    pattern1_flat = pattern1.reshape(-1)
    pattern2_flat = pattern2.reshape(-1)

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        pattern1_flat.unsqueeze(0), pattern2_flat.unsqueeze(0)
    )

    return similarity.item()
