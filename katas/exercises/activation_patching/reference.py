"""Implement activation patching - reference solution."""

import torch
from transformer_lens import HookedTransformer


def patch_residual_stream(
    model: HookedTransformer,
    clean_prompt: str,
    corrupted_prompt: str,
    layer: int,
    position: int,
) -> torch.Tensor:
    """Patch residual stream from clean run into corrupted run."""
    # Get clean activation
    clean_tokens = model.to_tokens(clean_prompt)
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    clean_activation = clean_cache[f"blocks.{layer}.hook_resid_post"][0, position, :]

    # Define patching hook
    def patch_hook(activation, hook):
        activation[0, position, :] = clean_activation
        return activation

    # Run corrupted with patch
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    hook_name = f"blocks.{layer}.hook_resid_post"
    patched_logits = model.run_with_hooks(
        corrupted_tokens, fwd_hooks=[(hook_name, patch_hook)]
    )

    return patched_logits
