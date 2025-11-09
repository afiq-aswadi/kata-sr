"""Run TransformerLens model with activation caching."""

import torch
from transformer_lens import HookedTransformer


def run_with_cache(model: HookedTransformer, prompt: str) -> tuple[torch.Tensor, dict]:
    """Run model with full activation caching.

    Args:
        model: HookedTransformer model
        prompt: text prompt to run

    Returns:
        tuple of (logits, cache) where cache contains all activations
    """
    # BLANK_START
    raise NotImplementedError(
        "Convert prompt to tokens and run with caching. "
        "Hint: use model.to_tokens() and model.run_with_cache()"
    )
    # BLANK_END
