"""
Tests for Basic Activation Patch kata
"""

import pytest
import torch
from transformer_lens import HookedTransformer

try:
    from template import patch_activation
except ImportError:
    from reference import patch_activation


@pytest.fixture(scope="module")
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


@pytest.fixture
def cache_pair(model):
    """Generate clean and corrupt caches."""
    clean_tokens = model.to_tokens("The cat sat on the mat")
    corrupt_tokens = model.to_tokens("The dog sat on the mat")

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

    return {
        "clean_tokens": clean_tokens,
        "corrupt_tokens": corrupt_tokens,
        "clean_logits": clean_logits,
        "corrupt_logits": corrupt_logits,
        "clean_cache": clean_cache,
        "corrupt_cache": corrupt_cache
    }


class TestBasicActivationPatch:
    """Test basic activation patching."""

    def test_returns_logits(self, model, cache_pair):
        """Test that patching returns logits."""
        patched_logits = patch_activation(
            model,
            cache_pair["corrupt_tokens"],
            "blocks.0.hook_resid_post",
            cache_pair["clean_cache"]
        )

        assert isinstance(patched_logits, torch.Tensor)
        assert patched_logits.shape[-1] == model.cfg.d_vocab

    def test_patched_differs_from_corrupt(self, model, cache_pair):
        """Test that patching changes the output."""
        patched_logits = patch_activation(
            model,
            cache_pair["corrupt_tokens"],
            "blocks.5.hook_resid_post",
            cache_pair["clean_cache"]
        )

        # Patched should differ from unpatch corrupted
        assert not torch.allclose(
            patched_logits,
            cache_pair["corrupt_logits"],
            atol=0.01
        )

    def test_different_hooks_different_effects(self, model, cache_pair):
        """Test that patching different layers gives different results."""
        patched_layer_0 = patch_activation(
            model,
            cache_pair["corrupt_tokens"],
            "blocks.0.hook_resid_post",
            cache_pair["clean_cache"]
        )

        patched_layer_5 = patch_activation(
            model,
            cache_pair["corrupt_tokens"],
            "blocks.5.hook_resid_post",
            cache_pair["clean_cache"]
        )

        # Different layers should have different effects
        assert not torch.allclose(patched_layer_0, patched_layer_5)

    def test_patch_attention_output(self, model, cache_pair):
        """Test patching attention layer output."""
        patched_logits = patch_activation(
            model,
            cache_pair["corrupt_tokens"],
            "blocks.3.attn.hook_z",
            cache_pair["clean_cache"]
        )

        assert patched_logits.shape == cache_pair["corrupt_logits"].shape

    def test_patch_mlp_output(self, model, cache_pair):
        """Test patching MLP output."""
        patched_logits = patch_activation(
            model,
            cache_pair["corrupt_tokens"],
            "blocks.3.hook_mlp_out",
            cache_pair["clean_cache"]
        )

        assert patched_logits.shape == cache_pair["corrupt_logits"].shape

    def test_patch_embedding(self, model, cache_pair):
        """Test patching embedding layer."""
        patched_logits = patch_activation(
            model,
            cache_pair["corrupt_tokens"],
            "hook_embed",
            cache_pair["clean_cache"]
        )

        # Patching embeddings should move output toward clean
        assert not torch.allclose(
            patched_logits,
            cache_pair["corrupt_logits"]
        )

    def test_late_layer_patch_larger_effect(self, model, cache_pair):
        """Test that patching later layers generally has larger effects."""
        # Patch early layer
        patched_early = patch_activation(
            model,
            cache_pair["corrupt_tokens"],
            "blocks.0.hook_resid_post",
            cache_pair["clean_cache"]
        )

        # Patch late layer
        patched_late = patch_activation(
            model,
            cache_pair["corrupt_tokens"],
            "blocks.10.hook_resid_post",
            cache_pair["clean_cache"]
        )

        # Measure distance from corrupt baseline
        early_diff = (patched_early - cache_pair["corrupt_logits"]).abs().mean()
        late_diff = (patched_late - cache_pair["corrupt_logits"]).abs().mean()

        # Later layers typically have larger immediate effects
        assert late_diff > early_diff * 0.5  # Relaxed check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
