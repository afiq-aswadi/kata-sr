"""Tests for ablate attention head kata."""

import pytest
import torch
from transformer_lens import HookedTransformer

try:
    from user_kata import ablate_attention_head
except ImportError:
    from .reference import ablate_attention_head


@pytest.fixture
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_output_shape(model):
    """Ablated output should have same shape as normal output"""
    text = "The cat sat on the mat"
    normal_logits = model(text)
    ablated_logits = ablate_attention_head(model, text, layer=0, head=0)

    assert normal_logits.shape == ablated_logits.shape
    model.reset_hooks()


def test_ablation_changes_output(model):
    """Ablating a head should change model output"""
    text = "The quick brown fox"
    normal_logits = model(text)
    ablated_logits = ablate_attention_head(model, text, layer=5, head=0)

    # Outputs should be different (ablation has an effect)
    assert not torch.allclose(normal_logits, ablated_logits, atol=1e-3)
    model.reset_hooks()


def test_ablate_different_heads(model):
    """Ablating different heads should produce different results"""
    text = "Testing different heads"
    ablated_head_0 = ablate_attention_head(model, text, layer=3, head=0)
    model.reset_hooks()
    ablated_head_1 = ablate_attention_head(model, text, layer=3, head=1)
    model.reset_hooks()

    # Different heads should have different effects
    assert not torch.allclose(ablated_head_0, ablated_head_1, atol=1e-3)


def test_ablate_different_layers(model):
    """Ablating heads in different layers should produce different results"""
    text = "Testing different layers"
    ablated_layer_0 = ablate_attention_head(model, text, layer=0, head=0)
    model.reset_hooks()
    ablated_layer_5 = ablate_attention_head(model, text, layer=5, head=0)
    model.reset_hooks()

    # Different layers should have different effects
    assert not torch.allclose(ablated_layer_0, ablated_layer_5, atol=1e-3)


def test_ablation_is_deterministic(model):
    """Same ablation should produce same output"""
    text = "Deterministic test"
    ablated_1 = ablate_attention_head(model, text, layer=2, head=3)
    model.reset_hooks()
    ablated_2 = ablate_attention_head(model, text, layer=2, head=3)
    model.reset_hooks()

    assert torch.allclose(ablated_1, ablated_2, atol=1e-6)


def test_hooks_cleaned_up(model):
    """Verify hooks don't persist after ablation"""
    text = "Hook cleanup test"

    # Run ablation
    ablate_attention_head(model, text, layer=0, head=0)

    # Reset hooks
    model.reset_hooks()

    # Normal forward pass should work without hooks
    normal_logits = model(text)
    assert normal_logits is not None


def test_single_token(model):
    """Should handle single token input"""
    text = "Hi"
    ablated_logits = ablate_attention_head(model, text, layer=0, head=0)

    assert ablated_logits.shape[1] == 1  # Single token
    model.reset_hooks()


def test_long_sequence(model):
    """Should handle longer sequences"""
    text = " ".join(["word"] * 20)
    ablated_logits = ablate_attention_head(model, text, layer=0, head=0)

    assert ablated_logits.shape[1] > 10  # Multiple tokens
    model.reset_hooks()
