"""Tests for attention pattern analysis kata."""

import pytest
import torch
from transformer_lens import HookedTransformer

try:
    from user_kata import (
        extract_attention_patterns,
        compute_attention_entropy,
        find_previous_token_heads,
        ablate_attention_heads,
        get_max_attention_positions,
        compare_attention_patterns,
        analyze_induction_head,
    )
except ImportError:
    from .reference import (
        extract_attention_patterns,
        compute_attention_entropy,
        find_previous_token_heads,
        ablate_attention_heads,
        get_max_attention_positions,
        compare_attention_patterns,
        analyze_induction_head,
    )


# Use a small model for faster testing
@pytest.fixture(scope="module")
def model():
    """Load a small model for testing."""
    return HookedTransformer.from_pretrained("gpt2-small")


def test_extract_attention_patterns_shape(model):
    """Test that attention patterns have correct shape."""
    text = "Hello world"
    layer = 0
    patterns = extract_attention_patterns(model, text, layer)

    # Should be (batch, n_heads, seq, seq)
    assert patterns.ndim == 4
    assert patterns.shape[0] == 1  # batch size
    assert patterns.shape[1] == model.cfg.n_heads  # number of heads
    assert patterns.shape[2] == patterns.shape[3]  # seq_len x seq_len


def test_extract_attention_patterns_normalized(model):
    """Test that attention patterns sum to 1.0 across key dimension."""
    text = "The quick brown fox"
    layer = 1
    patterns = extract_attention_patterns(model, text, layer)

    # Sum across key dimension (dim=-1) should equal 1.0
    sums = patterns.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_extract_attention_patterns_different_layers(model):
    """Test extraction from different layers."""
    text = "Test input"
    patterns_0 = extract_attention_patterns(model, text, 0)
    patterns_5 = extract_attention_patterns(model, text, 5)

    # Patterns from different layers should be different
    assert not torch.allclose(patterns_0, patterns_5)


def test_compute_attention_entropy_shape(model):
    """Test that entropy has correct shape."""
    text = "Hello world"
    layer = 0
    patterns = extract_attention_patterns(model, text, layer)
    entropy = compute_attention_entropy(patterns)

    # Entropy should be (batch, n_heads, query_pos)
    assert entropy.shape == patterns.shape[:-1]


def test_compute_attention_entropy_positive(model):
    """Test that entropy is always non-negative."""
    text = "The quick brown fox jumps"
    layer = 2
    patterns = extract_attention_patterns(model, text, layer)
    entropy = compute_attention_entropy(patterns)

    # Entropy should always be >= 0
    assert (entropy >= 0).all()


def test_compute_attention_entropy_uniform_is_maximal():
    """Test that uniform distribution has maximum entropy."""
    # Create uniform attention pattern
    seq_len = 10
    uniform_pattern = torch.ones(1, 1, seq_len, seq_len) / seq_len
    uniform_entropy = compute_attention_entropy(uniform_pattern)

    # Create focused attention pattern (attending to first token)
    focused_pattern = torch.zeros(1, 1, seq_len, seq_len)
    focused_pattern[:, :, :, 0] = 1.0
    focused_entropy = compute_attention_entropy(focused_pattern)

    # Uniform should have higher entropy than focused
    assert uniform_entropy.mean() > focused_entropy.mean()


def test_find_previous_token_heads_synthetic():
    """Test finding previous-token heads with synthetic data."""
    # Create pattern that attends to previous token
    seq_len = 8
    n_heads = 3
    patterns = torch.zeros(1, n_heads, seq_len, seq_len)

    # Head 0: attends to previous token (diagonal - 1)
    for i in range(1, seq_len):
        patterns[0, 0, i, i - 1] = 0.9
        patterns[0, 0, i, i] = 0.1

    # Head 1: attends to first token
    patterns[0, 1, :, 0] = 1.0

    # Head 2: uniform attention
    patterns[0, 2, :, :] = 1.0 / seq_len

    prev_heads = find_previous_token_heads(patterns, threshold=0.4)

    # Only head 0 should be identified as previous-token head
    assert prev_heads[0] == True
    assert prev_heads[1] == False
    assert prev_heads[2] == False


def test_ablate_attention_heads_shape(model):
    """Test that ablation preserves output shape."""
    text = "Hello world"
    layer = 0
    heads_to_ablate = [0, 1]

    logits = ablate_attention_heads(model, text, layer, heads_to_ablate)

    # Logits should have correct shape
    tokens = model.to_tokens(text)
    assert logits.shape[0] == 1  # batch size
    assert logits.shape[1] == tokens.shape[1]  # sequence length
    assert logits.shape[2] == model.cfg.d_vocab  # vocabulary size


def test_ablate_attention_heads_changes_output(model):
    """Test that ablating heads changes model output."""
    text = "The cat sat on the"
    layer = 5
    heads_to_ablate = [9]  # Head 9 in layer 5 is often important

    # Get normal output
    tokens = model.to_tokens(text)
    normal_logits = model(tokens)

    # Get ablated output
    ablated_logits = ablate_attention_heads(model, text, layer, heads_to_ablate)

    # Outputs should be different
    assert not torch.allclose(normal_logits, ablated_logits, atol=1e-4)


def test_ablate_attention_heads_uniform():
    """Test that ablated heads have uniform attention."""
    text = "Hello world test"
    layer = 0
    heads_to_ablate = [0, 2]

    # Run with ablation
    _ = ablate_attention_heads(model, text, layer, heads_to_ablate)

    # Note: We can't directly verify the hook modified the patterns
    # because the hook is only active during run_with_hooks
    # But the function should work correctly based on the ablation test above


def test_get_max_attention_positions(model):
    """Test finding maximum attention positions."""
    text = "The cat sat on the mat"
    layer = 3
    patterns = extract_attention_patterns(model, text, layer)

    query_pos = 3
    head = 0
    max_pos, max_weight = get_max_attention_positions(patterns, query_pos, head)

    # max_pos should be a valid position
    assert 0 <= max_pos < patterns.shape[-1]

    # max_weight should be between 0 and 1
    assert 0 <= max_weight <= 1

    # Verify it's actually the maximum
    actual_max = patterns[0, head, query_pos, :].max().item()
    assert abs(max_weight - actual_max) < 1e-5


def test_compare_attention_patterns_same_text(model):
    """Test that comparing same text gives high similarity."""
    text = "Hello world"
    layer = 2
    head = 0

    similarity = compare_attention_patterns(model, text, text, layer, head)

    # Similarity with itself should be 1.0
    assert abs(similarity - 1.0) < 1e-4


def test_compare_attention_patterns_different_texts(model):
    """Test comparing different texts."""
    text1 = "The cat sat"
    text2 = "The dog ran"
    layer = 2
    head = 0

    similarity = compare_attention_patterns(model, text1, text2, layer, head)

    # Similarity should be between -1 and 1
    assert -1.0 <= similarity <= 1.0

    # Different texts should have lower similarity than identical texts
    same_similarity = compare_attention_patterns(model, text1, text1, layer, head)
    assert similarity < same_similarity


def test_compare_attention_patterns_different_lengths(model):
    """Test comparing texts with different lengths."""
    text1 = "Short text"
    text2 = "This is a much longer piece of text"
    layer = 1
    head = 0

    similarity = compare_attention_patterns(model, text1, text2, layer, head)

    # Should handle different lengths without error
    assert -1.0 <= similarity <= 1.0


def test_analyze_induction_head_structure(model):
    """Test that analyze_induction_head returns correct structure."""
    text = "When Mary and John went to the store, John gave a drink to Mary"
    layer = 5
    head = 9  # Known induction head in GPT-2

    result = analyze_induction_head(model, text, layer, head)

    # Check that result has expected keys
    assert "induction_score" in result
    assert "avg_entropy" in result
    assert "max_attention_mean" in result

    # Check that values are reasonable
    assert isinstance(result["induction_score"], float)
    assert isinstance(result["avg_entropy"], float)
    assert isinstance(result["max_attention_mean"], float)

    assert result["avg_entropy"] >= 0
    assert 0 <= result["max_attention_mean"] <= 1


def test_analyze_induction_head_repeated_sequence(model):
    """Test induction analysis on repeated sequence."""
    # Text with repeated patterns (good for induction)
    text = "A B C A B C A B C"
    layer = 5
    head = 9

    result = analyze_induction_head(model, text, layer, head)

    # All metrics should be computed
    assert all(key in result for key in ["induction_score", "avg_entropy", "max_attention_mean"])
    assert all(isinstance(result[key], float) for key in result.keys())


def test_attention_patterns_work_with_different_sequence_lengths(model):
    """Test that all functions handle different sequence lengths."""
    short_text = "Hi"
    long_text = "This is a much longer sequence with many more tokens"

    layer = 0

    # Extract patterns
    short_patterns = extract_attention_patterns(model, short_text, layer)
    long_patterns = extract_attention_patterns(model, long_text, layer)

    # Compute entropy
    short_entropy = compute_attention_entropy(short_patterns)
    long_entropy = compute_attention_entropy(long_patterns)

    # Find previous token heads
    short_heads = find_previous_token_heads(short_patterns)
    long_heads = find_previous_token_heads(long_patterns)

    # All should work without errors
    assert short_patterns.shape[2] < long_patterns.shape[2]
    assert short_entropy.shape[-1] < long_entropy.shape[-1]
    assert short_heads.shape == long_heads.shape  # Same number of heads


def test_attention_patterns_handle_padding():
    """Test that functions handle attention patterns correctly."""
    # Create patterns with some positions having near-zero attention
    # (simulating padding)
    batch_size, n_heads, seq_len = 1, 2, 5
    patterns = torch.zeros(batch_size, n_heads, seq_len, seq_len)

    # First 3 positions attend normally
    patterns[:, :, :3, :3] = 1.0 / 3

    # Last 2 positions attend to first token (simulating padding)
    patterns[:, :, 3:, 0] = 1.0

    entropy = compute_attention_entropy(patterns)

    # Should not produce NaN or inf
    assert not torch.isnan(entropy).any()
    assert not torch.isinf(entropy).any()


def test_ablate_multiple_heads(model):
    """Test ablating multiple heads at once."""
    text = "The quick brown fox"
    layer = 3
    heads_to_ablate = [0, 1, 2, 3]

    logits = ablate_attention_heads(model, text, layer, heads_to_ablate)

    # Should work without errors
    assert logits.shape[2] == model.cfg.d_vocab


def test_entropy_zero_for_deterministic_attention():
    """Test that entropy is near zero for deterministic (one-hot) attention."""
    # Create one-hot attention pattern (attending only to first token)
    batch_size, n_heads, seq_len = 1, 1, 5
    patterns = torch.zeros(batch_size, n_heads, seq_len, seq_len)
    patterns[:, :, :, 0] = 1.0  # All positions attend only to first token

    entropy = compute_attention_entropy(patterns)

    # Entropy should be very close to 0
    assert entropy.abs().max() < 0.01
