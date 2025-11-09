"""Attention pattern analysis kata - reference solution."""

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
        attention patterns tensor of shape (batch, n_heads, query_pos, key_pos)
        These are post-softmax probabilities that sum to 1.0 across key dimension.
    """
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)
    patterns = cache[f"blocks.{layer}.attn.hook_pattern"]
    return patterns


def compute_attention_entropy(patterns: torch.Tensor) -> torch.Tensor:
    """Compute entropy of attention patterns for each query position.

    Entropy measures how "focused" vs "diffuse" attention is.
    High entropy = attention spread across many tokens
    Low entropy = attention focused on few tokens

    Args:
        patterns: attention patterns (batch, n_heads, query_pos, key_pos)

    Returns:
        entropy for each query position (batch, n_heads, query_pos)
        Formula: -sum(p * log(p)) for each query position
    """
    # Add small epsilon to avoid log(0)
    entropy = -(patterns * torch.log(patterns + 1e-10)).sum(dim=-1)
    return entropy


def find_previous_token_heads(
    patterns: torch.Tensor, threshold: float = 0.4
) -> torch.Tensor:
    """Identify attention heads that primarily attend to the previous token.

    Previous-token heads have high attention weight on the diagonal offset by 1.

    Args:
        patterns: attention patterns (batch, n_heads, query_pos, key_pos)
        threshold: minimum average attention to previous token to qualify

    Returns:
        boolean tensor (n_heads,) indicating which heads are previous-token heads
    """
    batch_size, n_heads, seq_len, _ = patterns.shape

    # Extract attention to previous token for positions 1 onwards
    # For position i (i >= 1), get patterns[:, :, i, i-1]
    prev_token_attn = torch.zeros(batch_size, n_heads, seq_len - 1)
    for i in range(1, seq_len):
        prev_token_attn[:, :, i - 1] = patterns[:, :, i, i - 1]

    # Average across batch and query positions
    avg_prev_attn = prev_token_attn.mean(dim=(0, 2))  # (n_heads,)

    return avg_prev_attn > threshold


def ablate_attention_heads(
    model: HookedTransformer,
    text: str,
    layer: int,
    heads: list[int],
) -> torch.Tensor:
    """Ablate specific attention heads by replacing with uniform distribution.

    This measures the causal impact of specific heads on model output.

    Args:
        model: HookedTransformer model
        text: input text
        layer: layer number
        heads: list of head indices to ablate

    Returns:
        model output logits with specified heads ablated
    """
    tokens = model.to_tokens(text)

    def ablate_hook(pattern, hook):
        seq_len = pattern.shape[-1]
        for head in heads:
            pattern[:, head, :, :] = 1.0 / seq_len
        return pattern

    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_pattern", ablate_hook)]
    )

    return logits


def get_max_attention_positions(
    patterns: torch.Tensor, query_pos: int, head: int, batch_idx: int = 0
) -> tuple[int, float]:
    """Find which position receives maximum attention from a query position.

    Args:
        patterns: attention patterns (batch, n_heads, query_pos, key_pos)
        query_pos: query position to analyze
        head: head index to analyze
        batch_idx: batch index

    Returns:
        tuple of (position receiving max attention, attention weight)
    """
    attn_weights = patterns[batch_idx, head, query_pos, :]
    max_pos = attn_weights.argmax().item()
    max_weight = attn_weights[max_pos].item()
    return max_pos, max_weight


def compare_attention_patterns(
    model: HookedTransformer,
    text1: str,
    text2: str,
    layer: int,
    head: int,
) -> float:
    """Compare attention patterns between two prompts using cosine similarity.

    Useful for understanding how attention changes with different inputs.

    Args:
        model: HookedTransformer model
        text1: first input text
        text2: second input text
        layer: layer number
        head: head index

    Returns:
        average cosine similarity across query positions
    """
    patterns1 = extract_attention_patterns(model, text1, layer)
    patterns2 = extract_attention_patterns(model, text2, layer)

    # Extract patterns for specific head
    pattern1 = patterns1[0, head]  # (seq1, seq1)
    pattern2 = patterns2[0, head]  # (seq2, seq2)

    # Handle different sequence lengths by using minimum
    min_seq_len = min(pattern1.shape[0], pattern2.shape[0])
    pattern1 = pattern1[:min_seq_len, :min_seq_len]
    pattern2 = pattern2[:min_seq_len, :min_seq_len]

    # Flatten patterns for cosine similarity
    pattern1_flat = pattern1.reshape(-1)
    pattern2_flat = pattern2.reshape(-1)

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        pattern1_flat.unsqueeze(0),
        pattern2_flat.unsqueeze(0)
    )

    return similarity.item()


def analyze_induction_head(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
) -> dict[str, float]:
    """Analyze if a head exhibits induction-like behavior.

    Induction heads attend to tokens that previously followed the current token.
    Classic pattern: "A B ... A" -> head attends to B when processing second A.

    Args:
        model: HookedTransformer model
        text: input text (should have repeated sequences)
        layer: layer number
        head: head index

    Returns:
        dict with metrics:
        - "induction_score": measure of induction behavior
        - "avg_entropy": average entropy of this head
        - "max_attention_mean": average of maximum attention weights
    """
    patterns = extract_attention_patterns(model, text, layer)
    head_pattern = patterns[0, head]  # (seq, seq)

    # Compute entropy
    entropy = compute_attention_entropy(patterns[0:1, head:head+1])
    avg_entropy = entropy.mean().item()

    # Compute average maximum attention weight
    max_weights = head_pattern.max(dim=-1)[0]  # max for each query position
    max_attention_mean = max_weights.mean().item()

    # Simple induction score: check if there's attention to offset positions
    # For a proper induction head, we'd look for attention at offset +1 from matching tokens
    # Here we use a simplified metric: average attention to positions that are not adjacent
    seq_len = head_pattern.shape[0]

    # Create a mask for non-adjacent positions (distance > 1)
    induction_score = 0.0
    if seq_len > 3:
        # Simple heuristic: attention to positions that are 2+ steps back
        far_attention = head_pattern[:, :-2].sum(dim=-1)  # Sum attention to non-recent positions
        induction_score = far_attention.mean().item()

    return {
        "induction_score": induction_score,
        "avg_entropy": avg_entropy,
        "max_attention_mean": max_attention_mean,
    }
