"""
Activation Cache Pair - Reference Implementation
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
    """
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

    return clean_logits, corrupt_logits, clean_cache, corrupt_cache
