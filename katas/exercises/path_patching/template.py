"""
Path Patching (Activation Patching) - Template

Implement activation patching to discover causal circuits in transformers.
Path patching helps identify which model components causally matter for a task
by swapping activations between clean and corrupted runs.

Key Concepts:
- Clean run: model run with correct/desired input
- Corrupted run: model run with modified/incorrect input
- Patching: replacing activation from corrupted run with clean run
- Logit difference: metric to measure model behavior
- Causal effect: how much does patching restore clean behavior?

Learning Goals:
1. Understand clean vs corrupted experimental setup
2. Implement activation patching with hooks
3. Compute patching metrics (normalized logit difference)
4. Systematically patch all heads/layers
5. Create patching heatmaps for circuit discovery
"""

import torch
from transformer_lens import HookedTransformer
from typing import Callable, Dict, Tuple, Optional
import einops


class PathPatcher:
    """
    Implements path patching for causal circuit discovery in transformers.
    """

    def __init__(self, model: HookedTransformer):
        self.model = model
        self.clean_cache = None
        self.corrupt_cache = None

    def run_with_cache_pair(
        self,
        clean_tokens: torch.Tensor,
        corrupt_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
        """
        Run both clean and corrupted inputs through the model, storing caches.

        Args:
            clean_tokens: Tokenized clean prompt [batch, seq_len]
            corrupt_tokens: Tokenized corrupted prompt [batch, seq_len]

        Returns:
            Tuple of (clean_logits, corrupt_logits, clean_cache, corrupt_cache)
        """
        # TODO: Implement cache pair generation
        # Hint: Use model.run_with_cache() for both inputs
        # Hint: Store results in self.clean_cache and self.corrupt_cache
        raise NotImplementedError("Implement run_with_cache_pair")

    def compute_logit_diff(
        self,
        logits: torch.Tensor,
        answer_token: int,
        wrong_token: int,
        position: int = -1
    ) -> torch.Tensor:
        """
        Compute logit difference between correct and incorrect answer.

        This is a common metric for measuring model behavior on tasks like
        Indirect Object Identification (IOI).

        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            answer_token: Token ID for correct answer
            wrong_token: Token ID for incorrect answer
            position: Which position to check (default: -1 for last token)

        Returns:
            Scalar tensor: logits[answer] - logits[wrong]
        """
        # TODO: Implement logit difference computation
        # Hint: Index logits at [batch=0, position, token_id]
        # Hint: Return difference between answer and wrong token logits
        raise NotImplementedError("Implement compute_logit_diff")

    def patch_activation(
        self,
        corrupt_tokens: torch.Tensor,
        hook_name: str,
        position: Optional[int] = None,
        head_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Patch a specific activation from clean cache into corrupted run.

        Args:
            corrupt_tokens: Tokens for corrupted input
            hook_name: Name of activation to patch (e.g., "blocks.5.hook_resid_post")
            position: If specified, only patch this position. If None, patch all.
            head_idx: If specified and patching attention, only patch this head.

        Returns:
            Patched logits
        """
        # TODO: Implement activation patching
        # Hint: Create a hook function that replaces activation with clean_cache value
        # Hint: Handle position indexing if position is specified
        # Hint: Handle head indexing if head_idx is specified (for attention outputs)
        # Hint: Use model.run_with_hooks() with fwd_hooks=[(hook_name, patch_fn)]
        raise NotImplementedError("Implement patch_activation")

    def compute_patching_effect(
        self,
        patched_logits: torch.Tensor,
        clean_logits: torch.Tensor,
        corrupt_logits: torch.Tensor,
        answer_token: int,
        wrong_token: int
    ) -> float:
        """
        Compute normalized patching effect.

        Returns value in [0, 1] where:
        - 0 = patching had no effect (still like corrupted)
        - 1 = patching fully restored clean behavior
        - Can be > 1 if patching overcorrects

        Formula: (patched - corrupt) / (clean - corrupt)

        Args:
            patched_logits: Logits after patching
            clean_logits: Logits from clean run
            corrupt_logits: Logits from corrupted run
            answer_token: Correct answer token ID
            wrong_token: Incorrect answer token ID

        Returns:
            Normalized patching effect
        """
        # TODO: Implement patching effect computation
        # Hint: Compute logit_diff for all three runs
        # Hint: Normalize: (patched - corrupt) / (clean - corrupt)
        # Hint: Handle division by zero if clean == corrupt
        raise NotImplementedError("Implement compute_patching_effect")

    def patch_head_path(
        self,
        corrupt_tokens: torch.Tensor,
        layer: int,
        head: int,
        position: Optional[int] = None
    ) -> torch.Tensor:
        """
        Patch a specific attention head's output (hook_z).

        Args:
            corrupt_tokens: Corrupted input tokens
            layer: Which layer (0 to n_layers-1)
            head: Which head (0 to n_heads-1)
            position: Optional position to patch

        Returns:
            Patched logits
        """
        # TODO: Implement head patching
        # Hint: Use hook name "blocks.{layer}.attn.hook_z"
        # Hint: hook_z has shape [batch, seq_len, n_heads, d_head]
        # Hint: Index with [:, :, head, :] to get specific head
        # Hint: If position specified, further index with [:, position, head, :]
        raise NotImplementedError("Implement patch_head_path")

    def patch_all_heads(
        self,
        clean_tokens: torch.Tensor,
        corrupt_tokens: torch.Tensor,
        answer_token: int,
        wrong_token: int,
        position: Optional[int] = None
    ) -> torch.Tensor:
        """
        Systematically patch all attention heads and measure effects.

        Args:
            clean_tokens: Clean prompt tokens
            corrupt_tokens: Corrupted prompt tokens
            answer_token: Correct answer token ID
            wrong_token: Incorrect answer token ID
            position: Optional position to restrict patching

        Returns:
            Tensor of shape [n_layers, n_heads] with patching effects
        """
        # TODO: Implement systematic head patching
        # Hint: First run clean and corrupt to get baseline logits
        # Hint: Create results tensor of zeros with shape [n_layers, n_heads]
        # Hint: Loop over all layers and heads
        # Hint: For each, call patch_head_path and compute_patching_effect
        # Hint: Store result in results[layer, head]
        raise NotImplementedError("Implement patch_all_heads")

    def patch_residual_stream(
        self,
        corrupt_tokens: torch.Tensor,
        layer: int,
        position: Optional[int] = None,
        stream_type: str = "post"
    ) -> torch.Tensor:
        """
        Patch the residual stream at a specific layer.

        Args:
            corrupt_tokens: Corrupted input tokens
            layer: Which layer to patch
            position: Optional specific position to patch
            stream_type: "pre" or "post" - before or after layer

        Returns:
            Patched logits
        """
        # TODO: Implement residual stream patching
        # Hint: Hook name is "blocks.{layer}.hook_resid_{stream_type}"
        # Hint: Use patch_activation with the appropriate hook name
        raise NotImplementedError("Implement patch_residual_stream")


def create_ioi_prompts() -> Tuple[str, str, int, int]:
    """
    Create a minimal pair for Indirect Object Identification task.

    Returns:
        (clean_prompt, corrupt_prompt, answer_token_id, wrong_token_id)
    """
    # TODO: Implement IOI prompt creation
    # Hint: Clean: "When Mary and John went to the store, John gave a drink to Mary"
    # Hint: Corrupt: "When Mary and John went to the store, John gave a drink to John"
    # Hint: Answer should be " Mary", wrong should be " John"
    # Hint: Use a model to tokenize and get token IDs
    raise NotImplementedError("Implement create_ioi_prompts")


def analyze_circuit(
    patching_results: torch.Tensor,
    threshold: float = 0.1
) -> Dict:
    """
    Analyze patching results to identify important circuit components.

    Args:
        patching_results: Tensor [n_layers, n_heads] of patching effects
        threshold: Minimum effect to consider component important

    Returns:
        Dictionary with:
        - important_heads: List of (layer, head) tuples above threshold
        - max_effect: Maximum patching effect found
        - max_head: (layer, head) with maximum effect
    """
    # TODO: Implement circuit analysis
    # Hint: Find where patching_results > threshold
    # Hint: Use torch.max to find maximum value and its location
    # Hint: Convert indices to (layer, head) tuples
    raise NotImplementedError("Implement analyze_circuit")
