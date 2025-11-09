"""Attention Pattern Analysis kata - reference solution."""

import torch
from transformer_lens import HookedTransformer


def extract_attention_patterns(
    model: HookedTransformer, text: str, layer: int
) -> torch.Tensor:
    """Extract attention patterns from a specific layer."""
    logits, cache = model.run_with_cache(text)
    return cache[f"blocks.{layer}.attn.hook_pattern"]


def compute_attention_entropy(patterns: torch.Tensor) -> torch.Tensor:
    """Compute entropy of attention patterns for each query position."""
    # Add small epsilon to prevent log(0)
    epsilon = 1e-10
    # Entropy = -sum(p * log(p))
    entropy = -(patterns * torch.log(patterns + epsilon)).sum(dim=-1)
    return entropy


def find_previous_token_heads(
    patterns: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Find attention heads that primarily attend to the previous token."""
    batch, n_heads, seq_len, _ = patterns.shape

    # For positions 1 onwards, get attention to previous token (diagonal offset by 1)
    if seq_len <= 1:
        return torch.zeros(n_heads, dtype=torch.bool)

    # Extract attention weights to previous token for each position
    prev_token_attn = []
    for i in range(1, seq_len):
        prev_token_attn.append(patterns[:, :, i, i-1])

    # Stack and average across positions and batch
    prev_token_attn = torch.stack(prev_token_attn, dim=-1)  # (batch, n_heads, seq-1)
    avg_prev_token_attn = prev_token_attn.mean(dim=(0, 2))  # (n_heads,)

    # Check which heads exceed threshold
    return avg_prev_token_attn > threshold


def ablate_attention_head(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Ablate (replace with uniform) a specific attention head's pattern."""

    def ablate_hook(pattern, hook):
        # Set the specified head to uniform distribution
        seq_len = pattern.shape[-1]
        pattern[:, head, :, :] = 1.0 / seq_len
        return pattern

    hook_name = f"blocks.{layer}.attn.hook_pattern"
    logits = model.run_with_hooks(text, fwd_hooks=[(hook_name, ablate_hook)])
    return logits


def compare_attention_patterns(
    model: HookedTransformer,
    text1: str,
    text2: str,
    layer: int,
    head: int,
) -> dict[str, torch.Tensor]:
    """Compare attention patterns for two different prompts."""
    # Extract patterns for both texts
    patterns1 = extract_attention_patterns(model, text1, layer)
    patterns2 = extract_attention_patterns(model, text2, layer)

    # Compute entropies
    entropy1 = compute_attention_entropy(patterns1)
    entropy2 = compute_attention_entropy(patterns2)

    # Extract specific head (remove batch dimension, select head)
    pattern1 = patterns1[0, head, :, :]  # (seq1, seq1)
    pattern2 = patterns2[0, head, :, :]  # (seq2, seq2)
    entropy1_head = entropy1[0, head, :]  # (seq1,)
    entropy2_head = entropy2[0, head, :]  # (seq2,)

    return {
        'pattern1': pattern1,
        'pattern2': pattern2,
        'entropy1': entropy1_head,
        'entropy2': entropy2_head,
    }


def get_max_attention_positions(
    patterns: torch.Tensor, top_k: int = 3
) -> torch.Tensor:
    """For each query position, find the top-k key positions it attends to most."""
    # torch.topk returns (values, indices)
    _, indices = torch.topk(patterns, k=top_k, dim=-1)
    return indices
