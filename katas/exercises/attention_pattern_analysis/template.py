"""Attention Pattern Analysis kata."""

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
        attention patterns tensor of shape (batch, n_heads, seq, seq)
        Post-softmax patterns (already normalized to sum to 1.0)
    """
    # TODO: run model with caching, extract attention patterns
    # Hint: use run_with_cache and access cache[f"blocks.{layer}.attn.hook_pattern"]
    # BLANK_START
    pass
    # BLANK_END


def compute_attention_entropy(patterns: torch.Tensor) -> torch.Tensor:
    """Compute entropy of attention patterns for each query position.

    Entropy measures how focused/diffuse the attention is.
    Low entropy = focused on few tokens, high entropy = spread across many.

    Args:
        patterns: attention patterns of shape (batch, n_heads, seq_q, seq_k)

    Returns:
        entropy tensor of shape (batch, n_heads, seq_q)
        Formula: -sum(p * log(p)) for each query position
    """
    # TODO: compute entropy for each query position
    # Hint: entropy = -sum(p * log(p + epsilon)) where epsilon prevents log(0)
    # Sum over the key dimension (last dimension)
    # BLANK_START
    pass
    # BLANK_END


def find_previous_token_heads(
    patterns: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Find attention heads that primarily attend to the previous token.

    A "previous token head" has high attention weight on the diagonal offset by 1.

    Args:
        patterns: attention patterns of shape (batch, n_heads, seq, seq)
        threshold: minimum average attention to previous token to qualify

    Returns:
        boolean tensor of shape (n_heads,) indicating which heads attend to previous token
    """
    # TODO: identify heads that attend to previous token
    # Hint: for each position i (starting from 1), check patterns[:, :, i, i-1]
    # Average across batch and positions, compare to threshold
    # BLANK_START
    pass
    # BLANK_END


def ablate_attention_head(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Ablate (replace with uniform) a specific attention head's pattern.

    This zeros out the head's contribution by making it attend uniformly.

    Args:
        model: HookedTransformer model
        text: input text
        layer: layer number
        head: head number to ablate

    Returns:
        model output logits with head ablated
    """
    # TODO: create hook that sets head's pattern to uniform distribution
    # Hint: pattern[:, head, :, :] = 1.0 / seq_len (uniform across keys)
    # Use run_with_hooks with hook on f"blocks.{layer}.attn.hook_pattern"
    # BLANK_START
    pass
    # BLANK_END


def compare_attention_patterns(
    model: HookedTransformer,
    text1: str,
    text2: str,
    layer: int,
    head: int,
) -> dict[str, torch.Tensor]:
    """Compare attention patterns for two different prompts.

    Args:
        model: HookedTransformer model
        text1: first input text
        text2: second input text
        layer: layer number
        head: head number to analyze

    Returns:
        dict with keys:
            'pattern1': attention pattern for text1, shape (seq1, seq1)
            'pattern2': attention pattern for text2, shape (seq2, seq2)
            'entropy1': entropy for text1, shape (seq1,)
            'entropy2': entropy for text2, shape (seq2,)
    """
    # TODO: extract patterns for both texts and compute their entropies
    # Hint: use extract_attention_patterns and compute_attention_entropy
    # Return patterns for the specific head (remove batch and head dimensions)
    # BLANK_START
    pass
    # BLANK_END


def get_max_attention_positions(
    patterns: torch.Tensor, top_k: int = 3
) -> torch.Tensor:
    """For each query position, find the top-k key positions it attends to most.

    Args:
        patterns: attention patterns of shape (batch, n_heads, seq_q, seq_k)
        top_k: number of top positions to return

    Returns:
        indices tensor of shape (batch, n_heads, seq_q, top_k)
        Contains the key positions with highest attention for each query
    """
    # TODO: find top-k attended positions for each query
    # Hint: use torch.topk on the last dimension
    # Return the indices of top-k values
    # BLANK_START
    pass
    # BLANK_END
