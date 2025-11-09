"""
Basic Activation Patch - Reference Implementation
"""

import torch
from transformer_lens import HookedTransformer
from typing import Dict


def patch_activation(
    model: HookedTransformer,
    corrupt_tokens: torch.Tensor,
    hook_name: str,
    clean_cache: Dict
) -> torch.Tensor:
    """
    Patch an activation from clean cache into corrupted forward pass.
    """
    def patch_hook(activation, hook):
        """Hook function that replaces activation with clean version."""
        return clean_cache[hook_name]

    patched_logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[(hook_name, patch_hook)]
    )

    return patched_logits
