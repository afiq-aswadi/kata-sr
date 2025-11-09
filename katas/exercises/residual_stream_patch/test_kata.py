"""
Tests for Residual Stream Patch kata
"""

import pytest
import torch
from transformer_lens import HookedTransformer

try:
    from template import patch_residual_stream
except ImportError:
    from reference import patch_residual_stream


@pytest.fixture(scope="module")
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


@pytest.fixture
def cache_pair(model):
    """Generate clean and corrupt caches."""
    clean_tokens = model.to_tokens("The cat sat")
    corrupt_tokens = model.to_tokens("The dog sat")

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

    return {
        "corrupt_tokens": corrupt_tokens,
        "clean_cache": clean_cache,
        "corrupt_logits": corrupt_logits
    }


class TestResidualStreamPatch:
    """Test residual stream patching."""

    def test_returns_logits(self, model, cache_pair):
        """Test that patching returns logits."""
        patched = patch_residual_stream(
            model, cache_pair["corrupt_tokens"], 0, cache_pair["clean_cache"]
        )

        assert patched.shape[-1] == model.cfg.d_vocab

    def test_post_stream_type(self, model, cache_pair):
        """Test patching post-layer residual stream."""
        patched = patch_residual_stream(
            model,
            cache_pair["corrupt_tokens"],
            layer=5,
            clean_cache=cache_pair["clean_cache"],
            stream_type="post"
        )

        assert patched.shape[-1] == model.cfg.d_vocab

    def test_pre_stream_type(self, model, cache_pair):
        """Test patching pre-layer residual stream."""
        patched = patch_residual_stream(
            model,
            cache_pair["corrupt_tokens"],
            layer=5,
            clean_cache=cache_pair["clean_cache"],
            stream_type="pre"
        )

        assert patched.shape[-1] == model.cfg.d_vocab

    def test_different_layers_different_effects(self, model, cache_pair):
        """Test that patching different layers gives different results."""
        patched_l0 = patch_residual_stream(
            model, cache_pair["corrupt_tokens"], 0, cache_pair["clean_cache"]
        )

        patched_l5 = patch_residual_stream(
            model, cache_pair["corrupt_tokens"], 5, cache_pair["clean_cache"]
        )

        assert not torch.allclose(patched_l0, patched_l5)

    def test_pre_vs_post_different(self, model, cache_pair):
        """Test that pre and post give different results."""
        patched_pre = patch_residual_stream(
            model,
            cache_pair["corrupt_tokens"],
            5,
            cache_pair["clean_cache"],
            stream_type="pre"
        )

        patched_post = patch_residual_stream(
            model,
            cache_pair["corrupt_tokens"],
            5,
            cache_pair["clean_cache"],
            stream_type="post"
        )

        assert not torch.allclose(patched_pre, patched_post)

    def test_changes_from_corrupt(self, model, cache_pair):
        """Test that patching changes output from corrupt baseline."""
        patched = patch_residual_stream(
            model, cache_pair["corrupt_tokens"], 3, cache_pair["clean_cache"]
        )

        assert not torch.allclose(
            patched, cache_pair["corrupt_logits"], atol=0.001
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
