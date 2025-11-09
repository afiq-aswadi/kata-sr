"""Tests for compare attention patterns kata."""

import pytest
import torch
from transformer_lens import HookedTransformer

try:
    from user_kata import compare_attention_patterns
except ImportError:
    from .reference import compare_attention_patterns


@pytest.fixture
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_output_keys(model):
    """Result should have all required keys"""
    result = compare_attention_patterns(model, "The cat", "The dog", layer=0, head=0)

    assert 'pattern1' in result
    assert 'pattern2' in result
    assert 'entropy1' in result
    assert 'entropy2' in result


def test_pattern_shapes(model):
    """Patterns should be 2D square matrices"""
    result = compare_attention_patterns(model, "Hello world", "Hi there", layer=0, head=0)

    # Patterns should be 2D
    assert result['pattern1'].dim() == 2
    assert result['pattern2'].dim() == 2

    # Square (seq, seq)
    assert result['pattern1'].shape[0] == result['pattern1'].shape[1]
    assert result['pattern2'].shape[0] == result['pattern2'].shape[1]


def test_entropy_shapes(model):
    """Entropies should be 1D vectors"""
    result = compare_attention_patterns(model, "Test one", "Test two", layer=1, head=2)

    # Entropies should be 1D
    assert result['entropy1'].dim() == 1
    assert result['entropy2'].dim() == 1

    # Length should match sequence length
    assert result['entropy1'].shape[0] == result['pattern1'].shape[0]
    assert result['entropy2'].shape[0] == result['pattern2'].shape[0]


def test_different_sequence_lengths(model):
    """Should handle texts with different lengths"""
    result = compare_attention_patterns(
        model, "Short", "This is a much longer sentence", layer=0, head=0
    )

    # Patterns can have different sizes
    assert result['pattern1'].shape[0] != result['pattern2'].shape[0]
    assert result['entropy1'].shape[0] != result['entropy2'].shape[0]


def test_same_text_gives_same_result(model):
    """Same text should give identical patterns and entropies"""
    text = "The quick brown fox"
    result = compare_attention_patterns(model, text, text, layer=2, head=1)

    assert torch.allclose(result['pattern1'], result['pattern2'])
    assert torch.allclose(result['entropy1'], result['entropy2'])


def test_different_heads(model):
    """Different heads should give different results"""
    text1, text2 = "Test", "Example"

    result_head0 = compare_attention_patterns(model, text1, text2, layer=0, head=0)
    result_head1 = compare_attention_patterns(model, text1, text2, layer=0, head=1)

    # Different heads should have different patterns
    assert not torch.allclose(result_head0['pattern1'], result_head1['pattern1'])


def test_different_layers(model):
    """Different layers should give different results"""
    text1, text2 = "Test", "Example"

    result_layer0 = compare_attention_patterns(model, text1, text2, layer=0, head=0)
    result_layer5 = compare_attention_patterns(model, text1, text2, layer=5, head=0)

    # Different layers should have different patterns
    assert not torch.allclose(result_layer0['pattern1'], result_layer5['pattern1'])


def test_patterns_normalized(model):
    """Patterns should be normalized (sum to 1.0)"""
    result = compare_attention_patterns(model, "The cat", "The dog", layer=0, head=0)

    # Each row should sum to 1.0
    sums1 = result['pattern1'].sum(dim=-1)
    sums2 = result['pattern2'].sum(dim=-1)

    assert torch.allclose(sums1, torch.ones_like(sums1), atol=1e-5)
    assert torch.allclose(sums2, torch.ones_like(sums2), atol=1e-5)


def test_entropy_non_negative(model):
    """Entropies should be non-negative"""
    result = compare_attention_patterns(model, "Test one", "Test two", layer=1, head=3)

    assert (result['entropy1'] >= 0).all()
    assert (result['entropy2'] >= 0).all()
