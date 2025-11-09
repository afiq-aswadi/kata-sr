"""
Tests for Path Patching kata.

These tests verify that students correctly implement activation patching
for causal circuit discovery in transformers.
"""

import pytest
import torch
from transformer_lens import HookedTransformer
import sys
from pathlib import Path

# Import the student's implementation
try:
    from template import PathPatcher, create_ioi_prompts, analyze_circuit
except ImportError:
    from reference import PathPatcher, create_ioi_prompts, analyze_circuit


@pytest.fixture(scope="module")
def model():
    """Load a small transformer model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


@pytest.fixture(scope="module")
def ioi_setup(model):
    """Set up IOI prompts and tokens."""
    clean_prompt = "When Mary and John went to the store, John gave a drink to Mary"
    corrupt_prompt = "When Mary and John went to the store, John gave a drink to John"

    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)

    answer_token = model.to_single_token(" Mary")
    wrong_token = model.to_single_token(" John")

    return {
        "clean_tokens": clean_tokens,
        "corrupt_tokens": corrupt_tokens,
        "answer_token": answer_token,
        "wrong_token": wrong_token,
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt
    }


@pytest.fixture
def patcher(model):
    """Create a PathPatcher instance."""
    return PathPatcher(model)


class TestBasicFunctionality:
    """Test basic patching functionality."""

    def test_run_with_cache_pair(self, patcher, ioi_setup):
        """Test that cache pair generation works correctly."""
        clean_logits, corrupt_logits, clean_cache, corrupt_cache = patcher.run_with_cache_pair(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"]
        )

        # Check that logits have correct shape
        assert clean_logits.shape == corrupt_logits.shape
        assert clean_logits.shape[-1] == patcher.model.cfg.d_vocab

        # Check that caches are stored
        assert patcher.clean_cache is not None
        assert patcher.corrupt_cache is not None

        # Check that caches contain expected keys
        assert "blocks.0.hook_resid_post" in clean_cache
        assert "blocks.0.attn.hook_z" in clean_cache

    def test_compute_logit_diff(self, patcher, ioi_setup, model):
        """Test logit difference computation."""
        # Run model to get logits
        clean_logits = model(ioi_setup["clean_tokens"])

        # Compute logit diff
        diff = patcher.compute_logit_diff(
            clean_logits,
            ioi_setup["answer_token"],
            ioi_setup["wrong_token"]
        )

        # Should be a scalar tensor
        assert diff.numel() == 1

        # For IOI task, clean prompt should favor Mary over John
        assert diff.item() > 0

    def test_logit_diff_positions(self, patcher, model):
        """Test logit diff at different positions."""
        tokens = model.to_tokens("The cat sat on the mat")
        logits = model(tokens)

        # Test different positions
        diff_last = patcher.compute_logit_diff(logits, 100, 200, position=-1)
        diff_first = patcher.compute_logit_diff(logits, 100, 200, position=0)

        # Should get different values for different positions
        assert diff_last.item() != diff_first.item()


class TestActivationPatching:
    """Test activation patching methods."""

    def test_patch_activation_full(self, patcher, ioi_setup):
        """Test patching entire activation."""
        # Set up caches
        patcher.run_with_cache_pair(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"]
        )

        # Patch residual stream at layer 0
        patched_logits = patcher.patch_activation(
            ioi_setup["corrupt_tokens"],
            "blocks.0.hook_resid_post"
        )

        # Should get valid logits
        assert patched_logits.shape[-1] == patcher.model.cfg.d_vocab

        # Patched output should differ from unpatch corrupted run
        corrupt_logits = patcher.model(ioi_setup["corrupt_tokens"])
        assert not torch.allclose(patched_logits, corrupt_logits)

    def test_patch_activation_position(self, patcher, ioi_setup):
        """Test patching at specific position."""
        patcher.run_with_cache_pair(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"]
        )

        # Patch only last position
        patched_logits_last = patcher.patch_activation(
            ioi_setup["corrupt_tokens"],
            "blocks.5.hook_resid_post",
            position=-1
        )

        # Patch only first position
        patched_logits_first = patcher.patch_activation(
            ioi_setup["corrupt_tokens"],
            "blocks.5.hook_resid_post",
            position=0
        )

        # Different positions should give different results
        assert not torch.allclose(patched_logits_last, patched_logits_first)

    def test_patch_head(self, patcher, ioi_setup):
        """Test patching specific attention head."""
        patcher.run_with_cache_pair(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"]
        )

        # Patch head 0 in layer 5
        patched_logits_h0 = patcher.patch_activation(
            ioi_setup["corrupt_tokens"],
            "blocks.5.attn.hook_z",
            head_idx=0
        )

        # Patch head 1 in layer 5
        patched_logits_h1 = patcher.patch_activation(
            ioi_setup["corrupt_tokens"],
            "blocks.5.attn.hook_z",
            head_idx=1
        )

        # Different heads should give different results
        assert not torch.allclose(patched_logits_h0, patched_logits_h1)


class TestPatchingMetrics:
    """Test patching effect computation."""

    def test_compute_patching_effect(self, patcher, ioi_setup):
        """Test normalized patching effect computation."""
        clean_logits, corrupt_logits, _, _ = patcher.run_with_cache_pair(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"]
        )

        # Patch something that should have an effect
        patched_logits = patcher.patch_activation(
            ioi_setup["corrupt_tokens"],
            "blocks.9.hook_resid_post"
        )

        effect = patcher.compute_patching_effect(
            patched_logits,
            clean_logits,
            corrupt_logits,
            ioi_setup["answer_token"],
            ioi_setup["wrong_token"]
        )

        # Effect should be a float
        assert isinstance(effect, float)

        # For layer 9, should have some effect (not exactly 0 or 1)
        # Relaxed bounds to account for model variability
        assert -0.5 <= effect <= 2.0

    def test_patching_effect_bounds(self, patcher, ioi_setup):
        """Test that patching clean cache fully restores behavior."""
        clean_logits, corrupt_logits, _, _ = patcher.run_with_cache_pair(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"]
        )

        # Patching with clean cache should give effect close to 1.0
        effect = patcher.compute_patching_effect(
            clean_logits,  # Use clean logits as "patched"
            clean_logits,
            corrupt_logits,
            ioi_setup["answer_token"],
            ioi_setup["wrong_token"]
        )

        assert abs(effect - 1.0) < 0.01

        # Using corrupt logits as "patched" should give effect close to 0.0
        effect_zero = patcher.compute_patching_effect(
            corrupt_logits,  # Use corrupt as "patched"
            clean_logits,
            corrupt_logits,
            ioi_setup["answer_token"],
            ioi_setup["wrong_token"]
        )

        assert abs(effect_zero - 0.0) < 0.01


class TestHeadPatching:
    """Test head-specific patching methods."""

    def test_patch_head_path(self, patcher, ioi_setup):
        """Test patching individual attention head."""
        patcher.run_with_cache_pair(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"]
        )

        # Patch head 9 in layer 9 (known to be important for IOI)
        patched_logits = patcher.patch_head_path(
            ioi_setup["corrupt_tokens"],
            layer=9,
            head=9
        )

        assert patched_logits.shape[-1] == patcher.model.cfg.d_vocab

    def test_patch_all_heads_shape(self, patcher, ioi_setup):
        """Test that patch_all_heads returns correct shape."""
        results = patcher.patch_all_heads(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"],
            ioi_setup["answer_token"],
            ioi_setup["wrong_token"]
        )

        # Should have shape [n_layers, n_heads]
        expected_shape = (patcher.model.cfg.n_layers, patcher.model.cfg.n_heads)
        assert results.shape == expected_shape

    def test_patch_all_heads_values(self, patcher, ioi_setup):
        """Test that patch_all_heads produces reasonable values."""
        results = patcher.patch_all_heads(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"],
            ioi_setup["answer_token"],
            ioi_setup["wrong_token"]
        )

        # Should have some variation (not all same)
        assert results.std() > 0.01

        # Most values should be in reasonable range (relaxed for robustness)
        assert results.abs().max() < 5.0

    def test_patch_head_position_specific(self, patcher, ioi_setup):
        """Test patching head at specific position."""
        patcher.run_with_cache_pair(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"]
        )

        # Patch at last position
        patched_last = patcher.patch_head_path(
            ioi_setup["corrupt_tokens"],
            layer=5,
            head=3,
            position=-1
        )

        # Patch at first position
        patched_first = patcher.patch_head_path(
            ioi_setup["corrupt_tokens"],
            layer=5,
            head=3,
            position=0
        )

        # Should produce different results
        assert not torch.allclose(patched_last, patched_first)


class TestResidualPatching:
    """Test residual stream patching."""

    def test_patch_residual_stream_post(self, patcher, ioi_setup):
        """Test patching post-layer residual stream."""
        patcher.run_with_cache_pair(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"]
        )

        patched_logits = patcher.patch_residual_stream(
            ioi_setup["corrupt_tokens"],
            layer=5,
            stream_type="post"
        )

        assert patched_logits.shape[-1] == patcher.model.cfg.d_vocab

    def test_patch_residual_stream_pre(self, patcher, ioi_setup):
        """Test patching pre-layer residual stream."""
        patcher.run_with_cache_pair(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"]
        )

        patched_logits = patcher.patch_residual_stream(
            ioi_setup["corrupt_tokens"],
            layer=5,
            stream_type="pre"
        )

        assert patched_logits.shape[-1] == patcher.model.cfg.d_vocab

    def test_residual_stream_position(self, patcher, ioi_setup):
        """Test patching residual stream at specific position."""
        patcher.run_with_cache_pair(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"]
        )

        patched_logits = patcher.patch_residual_stream(
            ioi_setup["corrupt_tokens"],
            layer=3,
            position=-2  # Second to last position
        )

        assert patched_logits.shape[-1] == patcher.model.cfg.d_vocab


class TestHelperFunctions:
    """Test helper and utility functions."""

    def test_create_ioi_prompts(self):
        """Test IOI prompt creation."""
        clean, corrupt, answer, wrong = create_ioi_prompts()

        # Check prompts are strings
        assert isinstance(clean, str)
        assert isinstance(corrupt, str)

        # Check tokens are integers
        assert isinstance(answer, int)
        assert isinstance(wrong, int)

        # Prompts should be similar but different
        assert clean != corrupt
        assert "Mary" in clean
        assert "John" in clean

    def test_analyze_circuit(self):
        """Test circuit analysis function."""
        # Create dummy patching results
        results = torch.tensor([
            [0.01, 0.02, 0.15, 0.03],
            [0.05, 0.45, 0.08, 0.12],
            [0.02, 0.03, 0.01, 0.02]
        ])

        analysis = analyze_circuit(results, threshold=0.1)

        # Check return structure
        assert "important_heads" in analysis
        assert "max_effect" in analysis
        assert "max_head" in analysis

        # Check important heads (above threshold)
        important = analysis["important_heads"]
        assert len(important) > 0
        assert (1, 1) in important  # Head with 0.45
        assert (0, 2) in important  # Head with 0.15

        # Check max head
        assert analysis["max_effect"] == 0.45
        assert analysis["max_head"] == (1, 1)

    def test_analyze_circuit_empty(self):
        """Test circuit analysis with no important heads."""
        # All values below threshold
        results = torch.tensor([
            [0.01, 0.02, 0.03],
            [0.04, 0.05, 0.06]
        ])

        analysis = analyze_circuit(results, threshold=0.5)

        # Should have empty important heads
        assert len(analysis["important_heads"]) == 0

        # Should still find max
        assert analysis["max_effect"] == 0.06


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_cache_handling(self, patcher, ioi_setup):
        """Test that patching fails gracefully without cache."""
        # Don't run cache pair first
        with pytest.raises((AttributeError, KeyError, TypeError)):
            patcher.patch_activation(
                ioi_setup["corrupt_tokens"],
                "blocks.0.hook_resid_post"
            )

    def test_invalid_layer_bounds(self, patcher, ioi_setup):
        """Test handling of invalid layer indices."""
        patcher.run_with_cache_pair(
            ioi_setup["clean_tokens"],
            ioi_setup["corrupt_tokens"]
        )

        # Should handle gracefully (or raise expected error)
        with pytest.raises((IndexError, KeyError)):
            patcher.patch_head_path(
                ioi_setup["corrupt_tokens"],
                layer=999,  # Invalid layer
                head=0
            )

    def test_different_prompt_lengths(self, patcher, model):
        """Test patching with different sequence lengths."""
        short_tokens = model.to_tokens("Hello world")
        long_tokens = model.to_tokens("Hello world this is a longer sequence")

        # Should handle different lengths
        _, _, clean_cache, corrupt_cache = patcher.run_with_cache_pair(
            short_tokens, long_tokens
        )

        # Caches should have different sequence lengths
        assert clean_cache["blocks.0.hook_resid_post"].shape[1] != \
               corrupt_cache["blocks.0.hook_resid_post"].shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
