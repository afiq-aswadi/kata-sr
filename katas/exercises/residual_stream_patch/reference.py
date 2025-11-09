"""
Residual Stream Patch - Reference Implementation
"""

import torch
from transformer_lens import HookedTransformer
from typing import Dict


def patch_residual_stream(
    model: HookedTransformer,
    corrupt_tokens: torch.Tensor,
    layer: int,
    clean_cache: Dict,
    stream_type: str = "post"
) -> torch.Tensor:
    """
    Patch the residual stream at a specific layer.
    """
    hook_name = f"blocks.{layer}.hook_resid_{stream_type}"

    def patch_hook(activation, hook):
        """Replace entire residual stream."""
        return clean_cache[hook_name]

    patched_logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[(hook_name, patch_hook)]
    )

    return patched_logits
