"""
Attention Head Patch

Patch a specific attention head's output to measure its causal importance.
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

    The hook_z activation has shape [batch, seq, n_heads, d_head].
    This function patches only the specified head, leaving others unchanged.

    Args:
        model: TransformerLens model
        corrupt_tokens: Corrupted input tokens [batch, seq_len]
        layer: Which layer (0 to n_layers-1)
        head: Which head (0 to n_heads-1)
        clean_cache: Cache from clean run

    Returns:
        Patched logits

    Example:
        >>> # Patch head 9 in layer 9 (important for IOI)
        >>> patched_logits = patch_attention_head(
        ...     model, corrupt_tokens, layer=9, head=9, clean_cache
        ... )
    """
    # BLANK_START
    raise NotImplementedError(
        "Create hook for 'blocks.{layer}.attn.hook_z' that replaces "
        "activation[:, :, head, :] with clean_cache value"
    )
    # BLANK_END
