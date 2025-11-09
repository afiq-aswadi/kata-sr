"""Tests for analyze_induction_head kata."""

import pytest
import torch
from transformer_lens import HookedTransformer

try:
    from user_kata import analyze_induction_head
except ImportError:
    from .reference import analyze_induction_head


@pytest.fixture(scope="module")
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_return_structure(model):
    """Should return dict with required keys."""
    result = analyze_induction_head(model, "A B C A B C", layer=5, head=9)

    assert isinstance(result, dict)
    assert "induction_score" in result
    assert "avg_entropy" in result
    assert "max_attention_mean" in result


def test_all_values_floats(model):
    """All values should be floats."""
    result = analyze_induction_head(model, "Test input", layer=3, head=0)

    assert isinstance(result["induction_score"], float)
    assert isinstance(result["avg_entropy"], float)
    assert isinstance(result["max_attention_mean"], float)


def test_entropy_non_negative(model):
    """Entropy should always be non-negative."""
    result = analyze_induction_head(model, "Hello world", layer=2, head=5)

    assert result["avg_entropy"] >= 0


def test_max_attention_in_range(model):
    """Max attention mean should be between 0 and 1."""
    result = analyze_induction_head(model, "The cat sat", layer=1, head=3)

    assert 0.0 <= result["max_attention_mean"] <= 1.0


def test_induction_score_valid(model):
    """Induction score should be a valid number."""
    result = analyze_induction_head(model, "A B C", layer=4, head=2)

    assert not torch.isnan(torch.tensor(result["induction_score"]))
    assert not torch.isinf(torch.tensor(result["induction_score"]))


def test_different_layers(model):
    """Can analyze different layers."""
    text = "Test"

    result_layer_0 = analyze_induction_head(model, text, layer=0, head=0)
    result_layer_5 = analyze_induction_head(model, text, layer=5, head=0)

    # Both should return valid results
    assert all(isinstance(v, float) for v in result_layer_0.values())
    assert all(isinstance(v, float) for v in result_layer_5.values())


def test_different_heads(model):
    """Can analyze different heads."""
    text = "Example"

    result_head_0 = analyze_induction_head(model, text, layer=3, head=0)
    result_head_9 = analyze_induction_head(model, text, layer=3, head=9)

    # Both should return valid results
    assert all(isinstance(v, float) for v in result_head_0.values())
    assert all(isinstance(v, float) for v in result_head_9.values())


def test_known_induction_head(model):
    """Known induction head (GPT-2 layer 5 head 9) should have metrics."""
    text = "When Mary and John went to the store, John gave a drink to Mary"
    result = analyze_induction_head(model, text, layer=5, head=9)

    # Should have all metrics computed
    assert result["avg_entropy"] > 0
    assert 0 < result["max_attention_mean"] <= 1.0


def test_repeated_sequence(model):
    """Works with repeated sequences (ideal for induction)."""
    text = "A B C A B C A B C"
    result = analyze_induction_head(model, text, layer=5, head=9)

    assert all(key in result for key in ["induction_score", "avg_entropy", "max_attention_mean"])


def test_short_sequence(model):
    """Handles short sequences."""
    text = "Hi"
    result = analyze_induction_head(model, text, layer=1, head=0)

    # Should still return valid metrics
    assert all(isinstance(v, float) for v in result.values())


def test_metrics_reasonable(model):
    """All metrics should be in reasonable ranges."""
    text = "The quick brown fox jumps over the lazy dog"
    result = analyze_induction_head(model, text, layer=4, head=6)

    # Entropy should be reasonable (not too high or too low for typical patterns)
    assert result["avg_entropy"] < 10  # Very high entropy unlikely

    # Max attention mean should be reasonable
    assert result["max_attention_mean"] > 0.0  # Should have some attention

    # All should be finite
    for value in result.values():
        assert not torch.isnan(torch.tensor(value))
        assert not torch.isinf(torch.tensor(value))
