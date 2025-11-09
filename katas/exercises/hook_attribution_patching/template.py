"""Attribution patching for mechanistic interpretability."""

import torch
from transformer_lens import HookedTransformer


def attribution_patching(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    hook_point: str,
) -> torch.Tensor:
    """Measure impact of patching clean → corrupted for each position.

    Args:
        model: HookedTransformer model
        clean_text: clean input text
        corrupted_text: corrupted input text
        hook_point: name of hook point to patch

    Returns:
        Impact scores for each position, shape (seq_len,)
    """
    # TODO:
    # 1. Get clean and corrupted baseline outputs
    # 2. For each position, patch clean activation into corrupted run
    # 3. Measure change in output (e.g., logit difference)
    # 4. Return impact scores for all positions
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
