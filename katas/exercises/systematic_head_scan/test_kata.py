"""
Tests for Systematic Head Scan kata
"""

import pytest
import torch
from transformer_lens import HookedTransformer

try:
    from template import scan_all_heads
except ImportError:
    from reference import scan_all_heads


@pytest.fixture(scope="module")
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


class TestSystematicHeadScan:
    """Test systematic head scanning."""

    def test_returns_correct_shape(self, model):
        """Test that result has shape [n_layers, n_heads]."""
        clean_tokens = model.to_tokens("The cat sat")
        corrupt_tokens = model.to_tokens("The dog sat")

        results = scan_all_heads(
            model, clean_tokens, corrupt_tokens, 100, 200
        )

        expected_shape = (model.cfg.n_layers, model.cfg.n_heads)
        assert results.shape == expected_shape

    def test_returns_tensor(self, model):
        """Test that result is a tensor."""
        clean_tokens = model.to_tokens("Hello")
        corrupt_tokens = model.to_tokens("World")

        results = scan_all_heads(
            model, clean_tokens, corrupt_tokens, 50, 75
        )

        assert isinstance(results, torch.Tensor)

    def test_contains_variation(self, model):
        """Test that not all heads have the same effect."""
        clean_tokens = model.to_tokens("The cat sat on")
        corrupt_tokens = model.to_tokens("The dog sat on")

        results = scan_all_heads(
            model, clean_tokens, corrupt_tokens, 100, 200
        )

        # Should have some variation across heads
        assert results.std() > 0.01

    def test_reasonable_values(self, model):
        """Test that values are in a reasonable range."""
        clean_tokens = model.to_tokens("Test prompt")
        corrupt_tokens = model.to_tokens("Test input")

        results = scan_all_heads(
            model, clean_tokens, corrupt_tokens, 50, 100
        )

        # Most values should be reasonable (not all huge or tiny)
        assert results.abs().max() < 10.0

    def test_different_prompts_different_results(self, model):
        """Test that different prompt pairs give different heatmaps."""
        # First pair
        clean1 = model.to_tokens("The cat")
        corrupt1 = model.to_tokens("The dog")
        results1 = scan_all_heads(model, clean1, corrupt1, 100, 200)

        # Second pair
        clean2 = model.to_tokens("Hello world")
        corrupt2 = model.to_tokens("Hello there")
        results2 = scan_all_heads(model, clean2, corrupt2, 100, 200)

        # Results should differ
        assert not torch.allclose(results1, results2, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
