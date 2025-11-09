"""Tests for TransformerLens position indexing kata."""

import pytest
import torch
from transformer_lens import HookedTransformer


@pytest.fixture
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_extract_position_returns_tensor(model):
    """Test that function returns a tensor."""
    from template import extract_position

    vec = extract_position(model, "Hello", layer=0, position=0)
    assert isinstance(vec, torch.Tensor)


def test_position_vector_shape(model):
    """Test that position vector has correct shape."""
    from template import extract_position

    vec = extract_position(model, "Test", layer=0, position=0)

    # Should be 1D vector of size d_model
    assert vec.dim() == 1
    assert vec.shape[0] == model.cfg.d_model


def test_different_positions_different_vectors(model):
    """Test that different positions have different activations."""
    from template import extract_position

    prompt = "The quick brown fox"
    pos0 = extract_position(model, prompt, layer=0, position=0)
    pos1 = extract_position(model, prompt, layer=0, position=1)
    pos2 = extract_position(model, prompt, layer=0, position=2)

    assert not torch.allclose(pos0, pos1)
    assert not torch.allclose(pos0, pos2)
    assert not torch.allclose(pos1, pos2)


def test_last_position_negative_indexing(model):
    """Test that -1 index extracts last token."""
    from template import extract_position

    prompt = "Test input here"
    last = extract_position(model, prompt, layer=0, position=-1)

    assert last.shape[0] == model.cfg.d_model


def test_different_layers_different_values(model):
    """Test that same position in different layers differs."""
    from template import extract_position

    prompt = "Test"
    layer0_pos0 = extract_position(model, prompt, layer=0, position=0)
    layer5_pos0 = extract_position(model, prompt, layer=5, position=0)

    assert not torch.allclose(layer0_pos0, layer5_pos0)


def test_position_values_finite(model):
    """Test that extracted values are finite."""
    from template import extract_position

    vec = extract_position(model, "Test", layer=0, position=0)
    assert torch.isfinite(vec).all()


def test_first_position_accessible(model):
    """Test that first position (0) can be extracted."""
    from template import extract_position

    vec = extract_position(model, "Hello world", layer=0, position=0)
    assert vec.shape[0] == model.cfg.d_model


def test_middle_position_accessible(model):
    """Test that middle positions can be extracted."""
    from template import extract_position

    prompt = "The quick brown fox jumps"
    mid_pos = 2
    vec = extract_position(model, prompt, layer=0, position=mid_pos)
    assert vec.shape[0] == model.cfg.d_model


def test_different_prompts_different_positions(model):
    """Test that same position index in different prompts differs."""
    from template import extract_position

    pos1 = extract_position(model, "Hello world", layer=0, position=0)
    pos2 = extract_position(model, "Goodbye world", layer=0, position=0)

    # Same position, different prompts -> different activations
    assert not torch.allclose(pos1, pos2)
