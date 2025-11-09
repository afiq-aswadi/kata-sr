"""
Activation Cache Pair

Generate and store activation caches for clean and corrupted model runs.
This is the foundation for activation patching experiments.
"""

import torch
from transformer_lens import HookedTransformer
from typing import Tuple, Dict


def run_with_cache_pair(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    """
    Run both clean and corrupted inputs, returning logits and caches.

    This function is essential for patching experiments - you need both
    the clean activations (to patch in) and corrupt activations (baseline).

    Args:
        model: TransformerLens model
        clean_tokens: Tokenized clean prompt [batch, seq_len]
        corrupt_tokens: Tokenized corrupted prompt [batch, seq_len]

    Returns:
        Tuple of (clean_logits, corrupt_logits, clean_cache, corrupt_cache)

    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> clean = model.to_tokens("The cat sat on the mat")
        >>> corrupt = model.to_tokens("The dog sat on the mat")
        >>> clean_logits, corrupt_logits, clean_cache, corrupt_cache = \\
        ...     run_with_cache_pair(model, clean, corrupt)
        >>> # Caches contain all intermediate activations
        >>> "blocks.0.hook_resid_post" in clean_cache
        True
    """
    # BLANK_START
    raise NotImplementedError(
        "Use model.run_with_cache() for both inputs and return all four values"
    )
    # BLANK_END
