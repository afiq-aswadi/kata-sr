"""
Path Patching (Activation Patching) - Reference Implementation

Complete implementation of activation patching for causal circuit discovery.
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
        """
        clean_logits, clean_cache = self.model.run_with_cache(clean_tokens)
        corrupt_logits, corrupt_cache = self.model.run_with_cache(corrupt_tokens)

        self.clean_cache = clean_cache
        self.corrupt_cache = corrupt_cache

        return clean_logits, corrupt_logits, clean_cache, corrupt_cache

    def compute_logit_diff(
        self,
        logits: torch.Tensor,
        answer_token: int,
        wrong_token: int,
        position: int = -1
    ) -> torch.Tensor:
        """
        Compute logit difference between correct and incorrect answer.
        """
        answer_logit = logits[0, position, answer_token]
        wrong_logit = logits[0, position, wrong_token]
        return answer_logit - wrong_logit

    def patch_activation(
        self,
        corrupt_tokens: torch.Tensor,
        hook_name: str,
        position: Optional[int] = None,
        head_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Patch a specific activation from clean cache into corrupted run.
        """
        def patch_hook(activation, hook):
            if position is not None and head_idx is not None:
                # Patch specific position and head (for attention)
                activation[:, position, head_idx, :] = self.clean_cache[hook_name][:, position, head_idx, :]
            elif position is not None:
                # Patch specific position only
                activation[:, position, :] = self.clean_cache[hook_name][:, position, :]
            elif head_idx is not None:
                # Patch specific head at all positions
                activation[:, :, head_idx, :] = self.clean_cache[hook_name][:, :, head_idx, :]
            else:
                # Patch entire activation
                activation[:] = self.clean_cache[hook_name][:]
            return activation

        patched_logits = self.model.run_with_hooks(
            corrupt_tokens,
            fwd_hooks=[(hook_name, patch_hook)]
        )

        return patched_logits

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
        """
        clean_diff = self.compute_logit_diff(clean_logits, answer_token, wrong_token)
        corrupt_diff = self.compute_logit_diff(corrupt_logits, answer_token, wrong_token)
        patched_diff = self.compute_logit_diff(patched_logits, answer_token, wrong_token)

        # Avoid division by zero
        denominator = clean_diff - corrupt_diff
        if abs(denominator) < 1e-6:
            return 0.0

        effect = (patched_diff - corrupt_diff) / denominator
        return effect.item()

    def patch_head_path(
        self,
        corrupt_tokens: torch.Tensor,
        layer: int,
        head: int,
        position: Optional[int] = None
    ) -> torch.Tensor:
        """
        Patch a specific attention head's output (hook_z).
        """
        hook_name = f"blocks.{layer}.attn.hook_z"
        return self.patch_activation(corrupt_tokens, hook_name, position, head)

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
        """
        # Get baseline logits
        clean_logits, corrupt_logits, _, _ = self.run_with_cache_pair(
            clean_tokens, corrupt_tokens
        )

        # Create results tensor
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        results = torch.zeros(n_layers, n_heads)

        # Patch each head
        for layer in range(n_layers):
            for head in range(n_heads):
                patched_logits = self.patch_head_path(
                    corrupt_tokens, layer, head, position
                )
                effect = self.compute_patching_effect(
                    patched_logits, clean_logits, corrupt_logits,
                    answer_token, wrong_token
                )
                results[layer, head] = effect

        return results

    def patch_residual_stream(
        self,
        corrupt_tokens: torch.Tensor,
        layer: int,
        position: Optional[int] = None,
        stream_type: str = "post"
    ) -> torch.Tensor:
        """
        Patch the residual stream at a specific layer.
        """
        hook_name = f"blocks.{layer}.hook_resid_{stream_type}"
        return self.patch_activation(corrupt_tokens, hook_name, position)


def create_ioi_prompts() -> Tuple[str, str, int, int]:
    """
    Create a minimal pair for Indirect Object Identification task.
    """
    clean_prompt = "When Mary and John went to the store, John gave a drink to Mary"
    corrupt_prompt = "When Mary and John went to the store, John gave a drink to John"

    # Use a simple model to get token IDs
    model = HookedTransformer.from_pretrained("gpt2-small")
    answer_token = model.to_single_token(" Mary")
    wrong_token = model.to_single_token(" John")

    return clean_prompt, corrupt_prompt, answer_token, wrong_token


def analyze_circuit(
    patching_results: torch.Tensor,
    threshold: float = 0.1
) -> Dict:
    """
    Analyze patching results to identify important circuit components.
    """
    # Find important heads (above threshold)
    important_mask = patching_results > threshold
    important_indices = torch.nonzero(important_mask)
    important_heads = [(idx[0].item(), idx[1].item()) for idx in important_indices]

    # Find maximum effect
    max_value = patching_results.max()
    max_idx = patching_results.argmax()
    n_heads = patching_results.shape[1]
    max_layer = max_idx // n_heads
    max_head = max_idx % n_heads

    return {
        "important_heads": important_heads,
        "max_effect": max_value.item(),
        "max_head": (max_layer.item(), max_head.item())
    }
