"""Run TransformerLens model with activation caching - reference solution."""

import torch
from transformer_lens import HookedTransformer


def run_with_cache(model: HookedTransformer, prompt: str) -> tuple[torch.Tensor, dict]:
    """Run model with full activation caching."""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    return logits, cache
