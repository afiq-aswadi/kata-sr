"""TransformerLens activation caching kata - reference solution."""

import torch
from transformer_lens import HookedTransformer


def load_model(model_name: str = "gpt2-small") -> HookedTransformer:
    """Load a HookedTransformer model."""
    return HookedTransformer.from_pretrained(model_name)


def run_with_full_cache(
    model: HookedTransformer, prompt: str
) -> tuple[torch.Tensor, dict]:
    """Run model with full activation caching."""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    return logits, cache


def extract_residual_stream(
    model: HookedTransformer, prompt: str, layer: int
) -> torch.Tensor:
    """Extract residual stream activations from a specific layer."""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    return cache[f"blocks.{layer}.hook_resid_post"]


def extract_attention_output(
    model: HookedTransformer, prompt: str, layer: int
) -> torch.Tensor:
    """Extract attention output (pre-projection) from a specific layer."""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    return cache[f"blocks.{layer}.attn.hook_z"]


def extract_mlp_output(
    model: HookedTransformer, prompt: str, layer: int
) -> torch.Tensor:
    """Extract MLP output from a specific layer."""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    return cache[f"blocks.{layer}.hook_mlp_out"]


def extract_last_token_residual(
    model: HookedTransformer, prompt: str, layer: int
) -> torch.Tensor:
    """Extract residual stream at the last token position."""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    residual = cache[f"blocks.{layer}.hook_resid_post"]
    return residual[0, -1, :]  # [batch=0, last_position, :]


def extract_specific_position(
    model: HookedTransformer, prompt: str, layer: int, position: int
) -> torch.Tensor:
    """Extract residual stream at a specific token position."""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    residual = cache[f"blocks.{layer}.hook_resid_post"]
    return residual[0, position, :]  # [batch=0, position, :]


def run_with_selective_cache(
    model: HookedTransformer, prompt: str, filter_fn
) -> tuple[torch.Tensor, dict]:
    """Run model with selective caching to save memory."""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens, names_filter=filter_fn)
    return logits, cache


def cache_only_residual_stream(
    model: HookedTransformer, prompt: str
) -> dict:
    """Cache only residual stream activations (not attention or MLP)."""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(
        tokens, names_filter=lambda name: "resid" in name
    )
    return cache


def compute_activation_stats(
    activations: torch.Tensor,
) -> dict[str, float]:
    """Compute statistics on cached activations."""
    return {
        "mean": activations.mean().item(),
        "std": activations.std().item(),
        "norm": activations.norm().item(),
    }
