"""Activation patching kata - reference solution."""

import torch
from transformer_lens import HookedTransformer


def patch_residual_stream(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    layer: int,
    position: int,
) -> torch.Tensor:
    """Patch residual stream from corrupted into clean run."""
    # Get corrupted activation
    _, corrupted_cache = model.run_with_cache(corrupted_text)
    corrupted_resid = corrupted_cache[f"blocks.{layer}.hook_resid_post"]

    # Hook to patch activation
    def patch_hook(activation, hook):
        activation[:, position, :] = corrupted_resid[:, position, :]
        return activation

    # Run clean with patched activation
    hook_name = f"blocks.{layer}.hook_resid_post"
    logits = model.run_with_hooks(clean_text, fwd_hooks=[(hook_name, patch_hook)])
    return logits


def patch_attention_head(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Patch attention head output from corrupted into clean run."""
    # Get corrupted head output
    _, corrupted_cache = model.run_with_cache(corrupted_text)
    corrupted_head = corrupted_cache[f"blocks.{layer}.attn.hook_z"]

    # Hook to patch specific head
    def patch_hook(activation, hook):
        activation[:, :, head, :] = corrupted_head[:, :, head, :]
        return activation

    # Run clean with patched head
    hook_name = f"blocks.{layer}.attn.hook_z"
    logits = model.run_with_hooks(clean_text, fwd_hooks=[(hook_name, patch_hook)])
    return logits


def compute_patching_effect(
    clean_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    target_token: int,
) -> float:
    """Compute how much patching recovered clean behavior."""
    clean_score = clean_logits[0, -1, target_token].item()
    corrupted_score = corrupted_logits[0, -1, target_token].item()
    patched_score = patched_logits[0, -1, target_token].item()

    if abs(clean_score - corrupted_score) < 1e-6:
        return 0.0

    return (patched_score - corrupted_score) / (clean_score - corrupted_score)


def scan_all_heads(
    model: HookedTransformer,
    clean_text: str,
    corrupted_text: str,
    target_token: int,
) -> torch.Tensor:
    """Scan all attention heads to find most important ones."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_logits = model(clean_text)
    corrupted_logits = model(corrupted_text)

    results = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        for head in range(n_heads):
            patched_logits = patch_attention_head(
                model, clean_text, corrupted_text, layer, head
            )
            effect = compute_patching_effect(
                clean_logits, corrupted_logits, patched_logits, target_token
            )
            results[layer, head] = effect

    return results
