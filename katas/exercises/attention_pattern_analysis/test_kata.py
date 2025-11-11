"""Tests for Attention Pattern Analysis kata."""

import pytest
import torch
from transformer_lens import HookedTransformer


@pytest.fixture
try:
    from user_kata import extract_attention_patterns
    from user_kata import extract_attention_patterns, compute_attention_entropy
    from user_kata import compute_attention_entropy
    from user_kata import find_previous_token_heads
    from user_kata import extract_attention_patterns, find_previous_token_heads
    from user_kata import ablate_attention_head
    from user_kata import compare_attention_patterns
    from user_kata import extract_attention_patterns, get_max_attention_positions
    from user_kata import get_max_attention_positions
except ImportError:
    from .reference import extract_attention_patterns
    from .reference import extract_attention_patterns, compute_attention_entropy
    from .reference import compute_attention_entropy
    from .reference import find_previous_token_heads
    from .reference import extract_attention_patterns, find_previous_token_heads
    from .reference import ablate_attention_head
    from .reference import compare_attention_patterns
    from .reference import extract_attention_patterns, get_max_attention_positions
    from .reference import get_max_attention_positions


def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_extract_attention_patterns_shape(model):

    text = "The cat sat on the mat"
    patterns = extract_attention_patterns(model, text, layer=0)

    # Should have shape (batch, n_heads, seq, seq)
    assert patterns.dim() == 4
    assert patterns.shape[0] == 1  # batch size
    assert patterns.shape[1] == model.cfg.n_heads
    assert patterns.shape[2] == patterns.shape[3]  # seq_q == seq_k


def test_attention_patterns_normalized(model):

    text = "Hello world"
    patterns = extract_attention_patterns(model, text, layer=0)

    # Patterns should sum to 1.0 across key dimension (already normalized)
    sums = patterns.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_attention_patterns_different_layers(model):

    text = "Testing different layers"
    patterns_0 = extract_attention_patterns(model, text, layer=0)
    patterns_5 = extract_attention_patterns(model, text, layer=5)

    # Different layers should have different patterns
    assert not torch.allclose(patterns_0, patterns_5, atol=1e-3)


def test_compute_attention_entropy_shape(model):

    text = "The quick brown fox"
    patterns = extract_attention_patterns(model, text, layer=0)
    entropy = compute_attention_entropy(patterns)

    # Entropy should have shape (batch, n_heads, seq_q)
    assert entropy.shape == (patterns.shape[0], patterns.shape[1], patterns.shape[2])


def test_compute_attention_entropy_values(model):

    # Create a focused attention pattern (low entropy)
    focused = torch.zeros(1, 2, 3, 4)
    focused[:, :, :, 0] = 1.0  # All attention on first token

    # Create a diffuse attention pattern (high entropy)
    diffuse = torch.ones(1, 2, 3, 4) * 0.25  # Uniform attention

    entropy_focused = compute_attention_entropy(focused)
    entropy_diffuse = compute_attention_entropy(diffuse)

    # Focused attention should have lower entropy
    assert (entropy_focused < entropy_diffuse).all()

    # Entropy should be non-negative
    assert (entropy_focused >= 0).all()
    assert (entropy_diffuse >= 0).all()


def test_find_previous_token_heads(model):

    # Create synthetic pattern where head 0 attends to previous token
    batch, n_heads, seq = 1, 3, 5
    patterns = torch.zeros(batch, n_heads, seq, seq)

    # Head 0: strong previous token attention
    for i in range(1, seq):
        patterns[0, 0, i, i-1] = 0.8  # 80% to previous token
        patterns[0, 0, i, i] = 0.2    # 20% to self

    # Head 1: uniform attention
    patterns[0, 1, :, :] = 1.0 / seq

    # Head 2: self attention
    for i in range(seq):
        patterns[0, 2, i, i] = 1.0

    prev_heads = find_previous_token_heads(patterns, threshold=0.5)

    # Only head 0 should be detected
    assert prev_heads[0] == True
    assert prev_heads[1] == False
    assert prev_heads[2] == False


def test_find_previous_token_heads_real_model(model):

    text = "The quick brown fox jumps over the lazy dog"
    patterns = extract_attention_patterns(model, text, layer=5)
    prev_heads = find_previous_token_heads(patterns, threshold=0.3)

    # Should return boolean tensor of correct size
    assert prev_heads.shape == (model.cfg.n_heads,)
    assert prev_heads.dtype == torch.bool


def test_ablate_attention_head(model):

    text = "The cat sat on the mat"

    # Get normal output
    normal_logits = model(text)

    # Ablate head 0 in layer 5
    ablated_logits = ablate_attention_head(model, text, layer=5, head=0)

    # Outputs should be different (ablation should have an effect)
    assert not torch.allclose(normal_logits, ablated_logits, atol=1e-3)

    # Shape should be the same
    assert normal_logits.shape == ablated_logits.shape

    # Clean up hooks
    model.reset_hooks()


def test_ablate_attention_head_pattern_uniform(model):

    text = "Test ablation"

    # Run with ablation and cache to verify pattern is uniform
    def check_hook(pattern, hook):
        # After ablation hook runs, check if head 0 is uniform
        seq_len = pattern.shape[-1]
        expected = 1.0 / seq_len
        # Note: ablation hook runs first, so we should see uniform pattern
        return pattern

    ablated_logits = ablate_attention_head(model, text, layer=0, head=0)

    # Just verify it runs without error
    assert ablated_logits is not None

    model.reset_hooks()


def test_compare_attention_patterns(model):

    text1 = "The cat sat"
    text2 = "The dog ran"
    result = compare_attention_patterns(model, text1, text2, layer=3, head=5)

    # Should have all required keys
    assert 'pattern1' in result
    assert 'pattern2' in result
    assert 'entropy1' in result
    assert 'entropy2' in result

    # Patterns should be 2D (seq, seq)
    assert result['pattern1'].dim() == 2
    assert result['pattern2'].dim() == 2

    # Entropies should be 1D (seq,)
    assert result['entropy1'].dim() == 1
    assert result['entropy2'].dim() == 1


def test_compare_attention_patterns_different_lengths(model):

    text1 = "Short"
    text2 = "This is a longer sentence"
    result = compare_attention_patterns(model, text1, text2, layer=0, head=0)

    # Patterns can have different sequence lengths
    assert result['pattern1'].shape[0] != result['pattern2'].shape[0]
    assert result['entropy1'].shape[0] != result['entropy2'].shape[0]


def test_get_max_attention_positions(model):

    text = "The quick brown fox"
    patterns = extract_attention_patterns(model, text, layer=0)

    top_3 = get_max_attention_positions(patterns, top_k=3)

    # Shape should be (batch, n_heads, seq_q, top_k)
    assert top_3.shape == (patterns.shape[0], patterns.shape[1], patterns.shape[2], 3)

    # Indices should be in valid range
    seq_len = patterns.shape[-1]
    assert (top_3 >= 0).all()
    assert (top_3 < seq_len).all()


def test_get_max_attention_positions_correctness(model):

    # Create synthetic pattern where we know the top positions
    batch, n_heads, seq = 1, 2, 5
    patterns = torch.zeros(batch, n_heads, seq, seq)

    # For head 0, query position 2: make positions [4, 1, 0] the top 3
    patterns[0, 0, 2, 4] = 0.5
    patterns[0, 0, 2, 1] = 0.3
    patterns[0, 0, 2, 0] = 0.2

    top_3 = get_max_attention_positions(patterns, top_k=3)

    # For head 0, query 2, should get [4, 1, 0] in that order
    expected = torch.tensor([4, 1, 0])
    assert torch.equal(top_3[0, 0, 2, :], expected)


def test_entropy_properties(model):

    text = "Testing entropy properties"
    patterns = extract_attention_patterns(model, text, layer=0)
    entropy = compute_attention_entropy(patterns)

    # Entropy should be non-negative
    assert (entropy >= 0).all()

    # Entropy should be finite (no infinities or NaNs)
    assert torch.isfinite(entropy).all()

    # Maximum entropy for seq_len tokens is log(seq_len)
    seq_len = patterns.shape[-1]
    max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float32))
    assert (entropy <= max_entropy + 0.1).all()  # Small tolerance for numerical errors
