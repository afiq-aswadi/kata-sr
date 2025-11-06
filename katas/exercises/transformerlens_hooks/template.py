"""TransformerLens hooks kata."""

import torch
from transformer_lens import HookedTransformer


def extract_attention_patterns(
    model: HookedTransformer, text: str, layer: int
) -> torch.Tensor:
    """Extract attention patterns from a specific layer.

    Args:
        model: HookedTransformer model
        text: input text
        layer: layer number to extract from

    Returns:
        attention patterns tensor of shape (batch, heads, seq, seq)
    """
    # TODO: run model with caching, extract attention patterns
    # Hint: use run_with_cache and access cache[f"blocks.{layer}.attn.hook_pattern"]
    # BLANK_START
    pass
    # BLANK_END


def get_residual_stream(
    model: HookedTransformer, text: str, layer: int, position: int
) -> torch.Tensor:
    """Get residual stream activations at specific layer and position.

    Args:
        model: HookedTransformer model
        text: input text
        layer: layer number
        position: token position

    Returns:
        residual stream vector at that position
    """
    # TODO: run model with cache, extract residual stream
    # Hint: cache[f"blocks.{layer}.hook_resid_post"]
    # BLANK_START
    pass
    # BLANK_END


def register_activation_hook(
    model: HookedTransformer, hook_point: str
) -> list[torch.Tensor]:
    """Register a hook to collect activations from a hook point.

    Args:
        model: HookedTransformer model
        hook_point: name of hook point (e.g., "blocks.0.attn.hook_z")

    Returns:
        list to collect activations (will be filled during forward pass)
    """
    activations = []

    def hook_fn(activation, hook):
        # TODO: append activation to list
        # BLANK_START
        pass
        # BLANK_END

    # TODO: register hook using model.add_hook()
    # BLANK_START
    pass
    # BLANK_END

    return activations


def mean_ablate_head(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Ablate (zero out) a specific attention head.

    Args:
        model: HookedTransformer model
        text: input text
        layer: layer number
        head: head number

    Returns:
        model output logits with head ablated
    """
    # TODO: create hook that sets head output to zero
    # Hint: hook should modify activation[:, :, head, :] = 0
    # BLANK_START
    pass
    # BLANK_END
