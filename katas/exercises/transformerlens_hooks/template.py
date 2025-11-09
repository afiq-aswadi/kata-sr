"""TransformerLens activation caching kata.

Master model loading and activation extraction for mechanistic interpretability.
"""

import torch
from transformer_lens import HookedTransformer


def load_model(model_name: str = "gpt2-small") -> HookedTransformer:
    """Load a HookedTransformer model.

    Args:
        model_name: name of model to load (e.g., "gpt2-small", "gpt2-medium")

    Returns:
        loaded HookedTransformer model
    """
    # TODO: Load model using HookedTransformer.from_pretrained()
    # BLANK_START
    pass
    # BLANK_END


def run_with_full_cache(
    model: HookedTransformer, prompt: str
) -> tuple[torch.Tensor, dict]:
    """Run model with full activation caching.

    Args:
        model: HookedTransformer model
        prompt: text prompt to run

    Returns:
        tuple of (logits, cache) where cache contains all activations
    """
    # TODO: Convert prompt to tokens and run with caching
    # Hint: use model.to_tokens() and model.run_with_cache()
    # BLANK_START
    pass
    # BLANK_END


def extract_residual_stream(
    model: HookedTransformer, prompt: str, layer: int
) -> torch.Tensor:
    """Extract residual stream activations from a specific layer.

    Args:
        model: HookedTransformer model
        prompt: text prompt
        layer: layer number (0 to n_layers-1)

    Returns:
        residual stream tensor of shape (batch, seq_len, d_model)
    """
    # TODO: run with cache and extract residual stream
    # Hint: cache key is f"blocks.{layer}.hook_resid_post"
    # BLANK_START
    pass
    # BLANK_END


def extract_attention_output(
    model: HookedTransformer, prompt: str, layer: int
) -> torch.Tensor:
    """Extract attention output (pre-projection) from a specific layer.

    Args:
        model: HookedTransformer model
        prompt: text prompt
        layer: layer number

    Returns:
        attention output tensor of shape (batch, seq_len, n_heads, d_head)
    """
    # TODO: extract attention output using cache[f"blocks.{layer}.attn.hook_z"]
    # BLANK_START
    pass
    # BLANK_END


def extract_mlp_output(
    model: HookedTransformer, prompt: str, layer: int
) -> torch.Tensor:
    """Extract MLP output from a specific layer.

    Args:
        model: HookedTransformer model
        prompt: text prompt
        layer: layer number

    Returns:
        MLP output tensor of shape (batch, seq_len, d_model)
    """
    # TODO: extract MLP output using cache[f"blocks.{layer}.hook_mlp_out"]
    # BLANK_START
    pass
    # BLANK_END


def extract_last_token_residual(
    model: HookedTransformer, prompt: str, layer: int
) -> torch.Tensor:
    """Extract residual stream at the last token position.

    Args:
        model: HookedTransformer model
        prompt: text prompt
        layer: layer number

    Returns:
        residual stream vector at last position, shape (d_model,)
    """
    # TODO: extract residual stream and index to get last token
    # Hint: use [:, -1, :] to get last position
    # BLANK_START
    pass
    # BLANK_END


def extract_specific_position(
    model: HookedTransformer, prompt: str, layer: int, position: int
) -> torch.Tensor:
    """Extract residual stream at a specific token position.

    Args:
        model: HookedTransformer model
        prompt: text prompt
        layer: layer number
        position: token position (0-indexed)

    Returns:
        residual stream vector at that position, shape (d_model,)
    """
    # TODO: extract and index to specific position
    # BLANK_START
    pass
    # BLANK_END


def run_with_selective_cache(
    model: HookedTransformer, prompt: str, filter_fn
) -> tuple[torch.Tensor, dict]:
    """Run model with selective caching to save memory.

    Args:
        model: HookedTransformer model
        prompt: text prompt
        filter_fn: function that takes activation name and returns True to cache it

    Returns:
        tuple of (logits, cache) with only filtered activations
    """
    # TODO: run with cache using names_filter parameter
    # Hint: model.run_with_cache(tokens, names_filter=filter_fn)
    # BLANK_START
    pass
    # BLANK_END


def cache_only_residual_stream(
    model: HookedTransformer, prompt: str
) -> dict:
    """Cache only residual stream activations (not attention or MLP).

    Args:
        model: HookedTransformer model
        prompt: text prompt

    Returns:
        cache containing only residual stream activations
    """
    # TODO: use names_filter to cache only activations with "resid" in name
    # Hint: lambda name: "resid" in name
    # BLANK_START
    pass
    # BLANK_END


def compute_activation_stats(
    activations: torch.Tensor,
) -> dict[str, float]:
    """Compute statistics on cached activations.

    Args:
        activations: activation tensor of any shape

    Returns:
        dict with keys: "mean", "std", "norm" (L2 norm)
    """
    # TODO: compute mean, std, and L2 norm of activations
    # Hint: use .mean(), .std(), .norm() methods
    # BLANK_START
    pass
    # BLANK_END
