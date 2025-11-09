"""
Tests for Activation Cache Pair kata
"""

import pytest
import torch
from transformer_lens import HookedTransformer

try:
    from template import run_with_cache_pair
except ImportError:
    from reference import run_with_cache_pair


@pytest.fixture(scope="module")
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


class TestActivationCachePair:
    """Test activation cache pair generation."""

    def test_returns_four_values(self, model):
        """Test that function returns tuple of 4 values."""
        clean_tokens = model.to_tokens("The cat sat")
        corrupt_tokens = model.to_tokens("The dog sat")

        result = run_with_cache_pair(model, clean_tokens, corrupt_tokens)

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_logits_shape(self, model):
        """Test that logits have correct shape."""
        clean_tokens = model.to_tokens("The cat sat")
        corrupt_tokens = model.to_tokens("The dog sat")

        clean_logits, corrupt_logits, _, _ = run_with_cache_pair(
            model, clean_tokens, corrupt_tokens
        )

        # Should have same shape
        assert clean_logits.shape == corrupt_logits.shape
        # Should be [batch, seq_len, vocab_size]
        assert clean_logits.shape[-1] == model.cfg.d_vocab

    def test_caches_are_dicts(self, model):
        """Test that caches are dictionaries."""
        clean_tokens = model.to_tokens("The cat")
        corrupt_tokens = model.to_tokens("The dog")

        _, _, clean_cache, corrupt_cache = run_with_cache_pair(
            model, clean_tokens, corrupt_tokens
        )

        assert isinstance(clean_cache, dict)
        assert isinstance(corrupt_cache, dict)

    def test_caches_contain_expected_keys(self, model):
        """Test that caches contain activation keys."""
        clean_tokens = model.to_tokens("Hello")
        corrupt_tokens = model.to_tokens("World")

        _, _, clean_cache, corrupt_cache = run_with_cache_pair(
            model, clean_tokens, corrupt_tokens
        )

        # Check for standard activation names
        expected_keys = [
            "blocks.0.hook_resid_post",
            "blocks.0.attn.hook_z",
            "hook_embed"
        ]

        for key in expected_keys:
            assert key in clean_cache, f"Missing {key} in clean cache"
            assert key in corrupt_cache, f"Missing {key} in corrupt cache"

    def test_different_prompts_different_logits(self, model):
        """Test that different inputs produce different logits."""
        clean_tokens = model.to_tokens("The cat sat")
        corrupt_tokens = model.to_tokens("The dog sat")

        clean_logits, corrupt_logits, _, _ = run_with_cache_pair(
            model, clean_tokens, corrupt_tokens
        )

        # Logits should be different for different inputs
        assert not torch.allclose(clean_logits, corrupt_logits)

    def test_different_prompts_different_caches(self, model):
        """Test that different inputs produce different cached activations."""
        clean_tokens = model.to_tokens("The cat sat")
        corrupt_tokens = model.to_tokens("The dog sat")

        _, _, clean_cache, corrupt_cache = run_with_cache_pair(
            model, clean_tokens, corrupt_tokens
        )

        # Check that residual stream activations differ
        clean_resid = clean_cache["blocks.5.hook_resid_post"]
        corrupt_resid = corrupt_cache["blocks.5.hook_resid_post"]

        assert not torch.allclose(clean_resid, corrupt_resid)

    def test_activation_shapes(self, model):
        """Test that cached activations have expected shapes."""
        clean_tokens = model.to_tokens("Test prompt")
        corrupt_tokens = model.to_tokens("Test prompt")

        _, _, clean_cache, _ = run_with_cache_pair(
            model, clean_tokens, corrupt_tokens
        )

        # Residual stream should be [batch, seq, d_model]
        resid = clean_cache["blocks.0.hook_resid_post"]
        assert resid.shape[-1] == model.cfg.d_model

        # Attention z should be [batch, seq, n_heads, d_head]
        attn_z = clean_cache["blocks.0.attn.hook_z"]
        assert attn_z.shape[2] == model.cfg.n_heads

    def test_same_prompt_gives_same_output(self, model):
        """Test deterministic behavior with same inputs."""
        tokens = model.to_tokens("Same prompt")

        result1 = run_with_cache_pair(model, tokens, tokens)
        result2 = run_with_cache_pair(model, tokens, tokens)

        # Clean and corrupt should be identical when using same prompt
        assert torch.allclose(result1[0], result1[1])
        assert torch.allclose(result2[0], result2[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
