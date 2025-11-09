"""Reference solution for patching activations."""

import torch
from transformer_lens import HookedTransformer


def hook_patch_activation(
    model: HookedTransformer,
    target_text: str,
    source_text: str,
    hook_point: str,
    position: int,
) -> torch.Tensor:
    """Patch activation from source_text into target_text run."""
    # Get source activation
    _, cache = model.run_with_cache(source_text)
    source_activation = cache[hook_point]

    # Define hook to patch
    def patch_hook(activation, hook):
        activation[:, position, :] = source_activation[:, position, :]
        return activation

    # Run target with patched activation
    logits = model.run_with_hooks(target_text, fwd_hooks=[(hook_point, patch_hook)])
    return logits
