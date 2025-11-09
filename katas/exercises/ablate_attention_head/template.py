"""Ablate attention head kata."""

import torch
from transformer_lens import HookedTransformer


def ablate_attention_head(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Ablate an attention head by replacing its pattern with uniform distribution.

    This zeros out the head's contribution by making it attend uniformly
    to all positions, removing any learned attention patterns.

    Args:
        model: HookedTransformer model
        text: Input text to process
        layer: Layer number containing the head
        head: Head number to ablate (0-indexed)

    Returns:
        Model logits with specified head ablated

    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> normal_logits = model("The cat")
        >>> ablated_logits = ablate_attention_head(model, "The cat", layer=5, head=9)
        >>> # ablated_logits will differ from normal_logits
    """
    # BLANK_START
    raise NotImplementedError("Create hook that sets pattern[:, head, :, :] = 1.0/seq_len, use run_with_hooks")
    # BLANK_END
