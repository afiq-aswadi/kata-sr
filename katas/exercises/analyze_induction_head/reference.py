"""Analyze if an attention head exhibits induction behavior."""

import torch
from transformer_lens import HookedTransformer


def analyze_induction_head(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
) -> dict[str, float]:
    """Analyze if a head exhibits induction-like behavior.

    Induction heads attend to tokens that previously followed the current token.
    Classic pattern: "A B ... A" -> head attends to B when processing second A.

    Args:
        model: HookedTransformer model
        text: input text (should have repeated sequences for best analysis)
        layer: layer number
        head: head index

    Returns:
        dict with three metrics:
        - "induction_score": measure of induction behavior
        - "avg_entropy": average entropy of this head's attention
        - "max_attention_mean": average of maximum attention weights
    """
    # Extract patterns for this layer
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)
    patterns = cache[f"blocks.{layer}.attn.hook_pattern"]

    head_pattern = patterns[0, head]  # (seq, seq)

    # Compute entropy
    entropy = -(patterns[0:1, head : head + 1] * torch.log(patterns[0:1, head : head + 1] + 1e-10)).sum(dim=-1)
    avg_entropy = entropy.mean().item()

    # Compute average maximum attention weight
    max_weights = head_pattern.max(dim=-1)[0]  # max for each query position
    max_attention_mean = max_weights.mean().item()

    # Simple induction score: attention to non-adjacent positions
    seq_len = head_pattern.shape[0]
    induction_score = 0.0
    if seq_len > 3:
        # Heuristic: attention to positions that are 2+ steps back
        far_attention = head_pattern[:, :-2].sum(dim=-1)
        induction_score = far_attention.mean().item()

    return {
        "induction_score": induction_score,
        "avg_entropy": avg_entropy,
        "max_attention_mean": max_attention_mean,
    }
