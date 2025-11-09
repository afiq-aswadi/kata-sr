"""Ablate attention head kata - reference solution."""

import torch
from transformer_lens import HookedTransformer


def ablate_attention_head(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Ablate an attention head by replacing its pattern with uniform distribution."""

    def ablate_hook(pattern, hook):
        # Set the specified head to uniform distribution
        seq_len = pattern.shape[-1]
        pattern[:, head, :, :] = 1.0 / seq_len
        return pattern

    hook_name = f"blocks.{layer}.attn.hook_pattern"
    logits = model.run_with_hooks(text, fwd_hooks=[(hook_name, ablate_hook)])
    return logits
