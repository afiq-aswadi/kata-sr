"""
Basic Activation Patch

Implement activation patching using TransformerLens hooks to swap
activations between clean and corrupted runs.
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

    This function replaces the activation at hook_name during a forward
    pass on corrupt_tokens with the corresponding activation from clean_cache.

    Args:
        model: TransformerLens model
        corrupt_tokens: Corrupted input tokens [batch, seq_len]
        hook_name: Name of activation to patch (e.g., "blocks.5.hook_resid_post")
        clean_cache: Cache from clean run containing activations

    Returns:
        Patched logits from the corrupted run with clean activation

    Example:
        >>> # First get caches
        >>> clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        >>> corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)
        >>>
        >>> # Patch layer 5 residual stream
        >>> patched_logits = patch_activation(
        ...     model, corrupt_tokens, "blocks.5.hook_resid_post", clean_cache
        ... )
        >>> # patched_logits will be closer to clean_logits if layer 5 is important
    """
    # BLANK_START
    raise NotImplementedError(
        "Create a hook function that replaces activation with clean_cache[hook_name], "
        "then use model.run_with_hooks(corrupt_tokens, fwd_hooks=[(hook_name, hook_fn)])"
    )
    # BLANK_END
