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
    """
    tokens = model.to_tokens(text)

    def ablate_hook(pattern, hook):
        seq_len = pattern.shape[-1]
        pattern[:, head, :, :] = 1.0 / seq_len
        return pattern

    logits = model.run_with_hooks(
        tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_pattern", ablate_hook)]
    )

    return logits
