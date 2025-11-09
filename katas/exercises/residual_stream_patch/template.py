"""
Residual Stream Patch

Patch the full residual stream at a layer boundary.
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

    Args:
        model: TransformerLens model
        corrupt_tokens: Corrupted input tokens [batch, seq_len]
        layer: Which layer to patch
        clean_cache: Cache from clean run
        stream_type: "pre" (before layer) or "post" (after layer)

    Returns:
        Patched logits

    Example:
        >>> # Patch residual stream after layer 5
        >>> patched = patch_residual_stream(
        ...     model, corrupt_tokens, layer=5, clean_cache=clean_cache
        ... )
    """
    # BLANK_START
    raise NotImplementedError(
        "Build hook name 'blocks.{layer}.hook_resid_{stream_type}' "
        "and patch full activation"
    )
    # BLANK_END
