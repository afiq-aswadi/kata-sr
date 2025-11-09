"""Ablate (zero out) activations in TransformerLens models."""

import torch
from transformer_lens import HookedTransformer


def hook_ablate_activation(
    model: HookedTransformer,
    text: str,
    hook_point: str,
    position: int,
) -> torch.Tensor:
    """Zero out activation at specific position.

    Args:
        model: HookedTransformer model
        text: input text
        hook_point: name of hook point (e.g., "blocks.0.hook_resid_post")
        position: token position to ablate

    Returns:
        Model output logits with activation ablated
    """
    # TODO: Define hook function that zeros out activation at position
    #       Use model.run_with_hooks to run with the hook
    # Hint: def hook_fn(activation, hook):
    #           activation[:, position, :] = 0
    #           return activation
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
