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

    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> metrics = analyze_induction_head(model, "A B C A B C", 5, 9)
        >>> "induction_score" in metrics
        True
    """
    # BLANK_START
    raise NotImplementedError(
        "Extract patterns, compute entropy, max attention, and induction score"
    )
    # BLANK_END
