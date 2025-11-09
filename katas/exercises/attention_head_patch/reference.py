"""
Attention Head Patch - Reference Implementation
"""

import torch
from transformer_lens import HookedTransformer
from typing import Dict


def patch_attention_head(
    model: HookedTransformer,
    corrupt_tokens: torch.Tensor,
    layer: int,
    head: int,
    clean_cache: Dict
) -> torch.Tensor:
    """
    Patch a specific attention head's output (hook_z).
    """
    hook_name = f"blocks.{layer}.attn.hook_z"

    def patch_hook(activation, hook):
        """Patch only the specified head."""
        activation[:, :, head, :] = clean_cache[hook_name][:, :, head, :]
        return activation

    patched_logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[(hook_name, patch_hook)]
    )

    return patched_logits
