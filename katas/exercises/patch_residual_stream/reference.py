"""Reference solution for patching residual stream."""

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer


def patch_residual_stream(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    layer: int,
    position: int,
) -> Float[torch.Tensor, "d_model"]:
    """Get direct effect on residual stream at position."""
    hook_point = f"blocks.{layer}.hook_resid_post"

    # Get clean activation
    _, clean_cache = model.run_with_cache(clean_text)
    clean_activation = clean_cache[hook_point]

    # Get corrupted baseline
    _, corrupted_cache = model.run_with_cache(corrupted_text)
    corrupted_activation = corrupted_cache[hook_point]

    # Patch and get new activation
    patched_cache = {}

    def patch_hook(activation, hook):
        activation[:, position, :] = clean_activation[:, position, :]
        patched_cache[hook_point] = activation.clone()
        return activation

    model.run_with_hooks(corrupted_text, fwd_hooks=[(hook_point, patch_hook)])

    # Return delta
    delta = patched_cache[hook_point][0, position, :] - corrupted_activation[0, position, :]
    return delta
