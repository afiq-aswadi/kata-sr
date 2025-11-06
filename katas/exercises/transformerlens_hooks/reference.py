"""TransformerLens hooks kata - reference solution."""

import torch
from transformer_lens import HookedTransformer


def extract_attention_patterns(
    model: HookedTransformer, text: str, layer: int
) -> torch.Tensor:
    """Extract attention patterns from a specific layer."""
    logits, cache = model.run_with_cache(text)
    return cache[f"blocks.{layer}.attn.hook_pattern"]


def get_residual_stream(
    model: HookedTransformer, text: str, layer: int, position: int
) -> torch.Tensor:
    """Get residual stream activations at specific layer and position."""
    logits, cache = model.run_with_cache(text)
    residual = cache[f"blocks.{layer}.hook_resid_post"]
    return residual[0, position, :]  # [batch=0, position, d_model]


def register_activation_hook(
    model: HookedTransformer, hook_point: str
) -> list[torch.Tensor]:
    """Register a hook to collect activations from a hook point."""
    activations = []

    def hook_fn(activation, hook):
        activations.append(activation.detach().clone())

    model.add_hook(hook_point, hook_fn)
    return activations


def mean_ablate_head(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Ablate (zero out) a specific attention head."""

    def ablate_hook(activation, hook):
        activation[:, :, head, :] = 0.0
        return activation

    hook_name = f"blocks.{layer}.attn.hook_z"
    logits = model.run_with_hooks(text, fwd_hooks=[(hook_name, ablate_hook)])
    return logits
