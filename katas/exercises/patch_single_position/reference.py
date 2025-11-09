"""Reference solution for patching single position."""

import torch
from transformer_lens import HookedTransformer


def patch_position(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    hook_point: str,
    position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch single token position, return (clean_output, patched_output)."""
    # Run clean and cache
    clean_logits, cache = model.run_with_cache(clean_text)
    clean_activation = cache[hook_point]

    # Patch hook
    def patch_hook(activation, hook):
        activation[:, position, :] = clean_activation[:, position, :]
        return activation

    # Run corrupted with patch
    patched_logits = model.run_with_hooks(
        corrupted_text, fwd_hooks=[(hook_point, patch_hook)]
    )

    return clean_logits, patched_logits
