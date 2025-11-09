"""Ablate attention head by replacing with uniform distribution."""

import torch
from transformer_lens import HookedTransformer


def ablate_attention_head(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Ablate a specific attention head by replacing with uniform distribution.

    This measures the causal impact of a specific head on model output.

    Args:
        model: HookedTransformer model
        text: input text to process
        layer: layer number containing the head
        head: head index to ablate (0-indexed)

    Returns:
        model output logits with specified head ablated

    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> logits = ablate_attention_head(model, "Hello world", layer=5, head=9)
        >>> logits.shape  # (batch, seq, vocab)
        torch.Size([1, 3, 50257])
    """
    # BLANK_START
    raise NotImplementedError(
        "Create hook that sets pattern[:, head, :, :] = 1.0/seq_len, use run_with_hooks"
    )
    # BLANK_END
