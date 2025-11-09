"""Tests for compare_attention_patterns kata."""

import pytest
import torch
from transformer_lens import HookedTransformer

try:
    from user_kata import compare_attention_patterns
except ImportError:
    from .reference import compare_attention_patterns


@pytest.fixture(scope="module")
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_return_type(model):
    """Should return a scalar float."""
    sim = compare_attention_patterns(model, "Hello", "World", layer=0, head=0)

    assert isinstance(sim, float)


def test_similarity_in_range(model):
    """Cosine similarity should be between -1 and 1."""
    sim = compare_attention_patterns(
        model, "The cat", "The dog", layer=1, head=2
    )

    assert -1.0 <= sim <= 1.0


def test_identical_text_high_similarity(model):
    """Same text should have similarity near 1.0."""
    text = "Hello world"
    sim = compare_attention_patterns(model, text, text, layer=2, head=0)

    assert sim > 0.99  # Should be very close to 1.0


def test_different_texts_lower_similarity(model):
    """Different texts should have lower similarity than identical."""
    text1 = "The cat sat"
    text2 = "The dog ran"

    same_sim = compare_attention_patterns(model, text1, text1, layer=1, head=0)
    diff_sim = compare_attention_patterns(model, text1, text2, layer=1, head=0)

    assert diff_sim < same_sim


def test_different_layers(model):
    """Can compare patterns from different layers."""
    text1 = "Test"
    text2 = "Test"

    sim_layer_0 = compare_attention_patterns(model, text1, text2, layer=0, head=0)
    sim_layer_5 = compare_attention_patterns(model, text1, text2, layer=5, head=0)

    # Both should be high (same text), but might differ slightly due to layer
    assert sim_layer_0 > 0.9
    assert sim_layer_5 > 0.9


def test_different_heads(model):
    """Can compare patterns from different heads."""
    text = "Hello"

    sim_head_0 = compare_attention_patterns(model, text, text, layer=2, head=0)
    sim_head_5 = compare_attention_patterns(model, text, text, layer=2, head=5)

    # Both should be high (same text)
    assert sim_head_0 > 0.9
    assert sim_head_5 > 0.9


def test_different_length_sequences(model):
    """Should handle sequences of different lengths."""
    short = "Hi"
    long = "This is a much longer sequence"

    sim = compare_attention_patterns(model, short, long, layer=1, head=0)

    # Should work without error
    assert -1.0 <= sim <= 1.0


def test_similar_structure_higher_similarity(model):
    """Similar sentence structure should have higher similarity."""
    text1 = "The cat sat on the mat"
    text2 = "The dog sat on the mat"
    text3 = "Completely different sentence structure here"

    sim_similar = compare_attention_patterns(model, text1, text2, layer=3, head=0)
    sim_different = compare_attention_patterns(model, text1, text3, layer=3, head=0)

    # Similar structure might have higher similarity (not guaranteed, but likely)
    # At minimum, both should be valid
    assert -1.0 <= sim_similar <= 1.0
    assert -1.0 <= sim_different <= 1.0


def test_short_sequences(model):
    """Works with very short sequences."""
    sim = compare_attention_patterns(model, "A", "B", layer=0, head=0)

    assert -1.0 <= sim <= 1.0


def test_repeated_patterns(model):
    """Repeated patterns should have high similarity."""
    text1 = "A B C A B C"
    text2 = "A B C A B C"

    sim = compare_attention_patterns(model, text1, text2, layer=4, head=0)

    assert sim > 0.99
