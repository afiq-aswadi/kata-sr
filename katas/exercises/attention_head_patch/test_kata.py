"""
Tests for Attention Head Patch kata
"""

import pytest
import torch
from transformer_lens import HookedTransformer

try:
except ImportError:
    from reference import patch_attention_head


@pytest.fixture(scope="module")
try:
    from user_kata import patch_attention_head
except ImportError:
    from .reference import patch_attention_head


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
        "clean_tokens": clean_tokens,
        "corrupt_tokens": corrupt_tokens,
        "clean_cache": clean_cache,
        "corrupt_cache": corrupt_cache,
        "corrupt_logits": corrupt_logits
    }


class TestAttentionHeadPatch:
    """Test attention head patching."""

    def test_returns_logits(self, model, cache_pair):
        """Test that patching returns logits."""
        patched = patch_attention_head(
            model, cache_pair["corrupt_tokens"], 0, 0, cache_pair["clean_cache"]
        )

        assert patched.shape[-1] == model.cfg.d_vocab

    def test_different_heads_different_effects(self, model, cache_pair):
        """Test that patching different heads gives different results."""
        patched_h0 = patch_attention_head(
            model, cache_pair["corrupt_tokens"], 5, 0, cache_pair["clean_cache"]
        )

        patched_h1 = patch_attention_head(
            model, cache_pair["corrupt_tokens"], 5, 1, cache_pair["clean_cache"]
        )

        assert not torch.allclose(patched_h0, patched_h1)

    def test_different_layers_different_effects(self, model, cache_pair):
        """Test that patching different layers gives different results."""
        patched_l0 = patch_attention_head(
            model, cache_pair["corrupt_tokens"], 0, 0, cache_pair["clean_cache"]
        )

        patched_l5 = patch_attention_head(
            model, cache_pair["corrupt_tokens"], 5, 0, cache_pair["clean_cache"]
        )

        assert not torch.allclose(patched_l0, patched_l5)

    def test_patch_changes_output(self, model, cache_pair):
        """Test that patching changes the output from corrupt baseline."""
        patched = patch_attention_head(
            model, cache_pair["corrupt_tokens"], 3, 5, cache_pair["clean_cache"]
        )

        assert not torch.allclose(
            patched, cache_pair["corrupt_logits"], atol=0.001
        )

    def test_all_heads_valid(self, model, cache_pair):
        """Test patching all heads in a layer."""
        layer = 3
        for head in range(model.cfg.n_heads):
            patched = patch_attention_head(
                model,
                cache_pair["corrupt_tokens"],
                layer,
                head,
                cache_pair["clean_cache"]
            )
            assert patched.shape[-1] == model.cfg.d_vocab


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
