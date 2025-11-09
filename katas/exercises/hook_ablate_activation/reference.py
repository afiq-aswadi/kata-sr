"""Reference solution for ablating activations."""

import torch
from transformer_lens import HookedTransformer


def hook_ablate_activation(
    model: HookedTransformer,
    text: str,
    hook_point: str,
    position: int,
) -> torch.Tensor:
    """Zero out activation at specific position."""
    def hook_fn(activation, hook):
        activation[:, position, :] = 0
        return activation

    logits = model.run_with_hooks(text, fwd_hooks=[(hook_point, hook_fn)])
    return logits
