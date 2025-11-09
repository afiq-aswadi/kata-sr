"""Tests for compute_attribution_scores kata."""

import torch
from transformer_lens import HookedTransformer

from framework import assert_shape

try:
    from user_kata import compute_attribution_scores
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    compute_attribution_scores = reference.compute_attribution_scores  # type: ignore


def test_output_structure():
    """Output should contain scores for each layer."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "The cat"
    corrupted_text = "The dog"

    def metric_fn(logits):
        return logits[0, -1, 0].item()  # Simple metric: first logit of last token

    scores = compute_attribution_scores(model, clean_text, corrupted_text, metric_fn)

    # Should have entry for each layer
    assert len(scores) == model.cfg.n_layers, f"should have {model.cfg.n_layers} layers"
    for layer in range(model.cfg.n_layers):
        assert f"layer_{layer}" in scores, f"should have layer_{layer}"


def test_score_shapes():
    """Each layer's scores should have shape (seq_len,)."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "Hello world"
    corrupted_text = "Goodbye moon"

    def metric_fn(logits):
        return logits[0, -1, 0].item()

    scores = compute_attribution_scores(model, clean_text, corrupted_text, metric_fn)

    tokens = model.to_tokens(clean_text)
    seq_len = tokens.shape[1]

    for layer in range(model.cfg.n_layers):
        layer_scores = scores[f"layer_{layer}"]
        assert_shape(layer_scores, (seq_len,), f"layer {layer} scores")


def test_some_positions_important():
    """At least some positions should have non-trivial scores."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "The cat sat on mat"
    corrupted_text = "The dog ran in park"

    def metric_fn(logits):
        # Metric: probability of token " cat" vs " dog"
        cat_token = model.to_single_token(" cat")
        dog_token = model.to_single_token(" dog")
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        return (probs[cat_token] - probs[dog_token]).item()

    scores = compute_attribution_scores(model, clean_text, corrupted_text, metric_fn)

    # At least one layer should have meaningful scores
    has_meaningful_scores = False
    for layer_scores in scores.values():
        if (layer_scores.abs() > 0.1).any():
            has_meaningful_scores = True
            break

    assert has_meaningful_scores, "at least some positions should have meaningful impact"


def test_different_metrics():
    """Should work with different metric functions."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "Test"
    corrupted_text = "Other"

    metrics = [
        lambda logits: logits[0, -1, 0].item(),
        lambda logits: logits[0, -1, :].sum().item(),
        lambda logits: logits[0, -1, :].max().item(),
    ]

    tokens = model.to_tokens(clean_text)
    seq_len = tokens.shape[1]

    for metric_fn in metrics:
        scores = compute_attribution_scores(model, clean_text, corrupted_text, metric_fn)
        assert len(scores) == model.cfg.n_layers
        for layer in range(model.cfg.n_layers):
            assert_shape(scores[f"layer_{layer}"], (seq_len,), f"metric variant layer {layer}")


def test_no_nans_or_infs():
    """Scores should not contain NaNs or infinities."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    clean_text = "Hello"
    corrupted_text = "World"

    def metric_fn(logits):
        return logits[0, -1, 0].item()

    scores = compute_attribution_scores(model, clean_text, corrupted_text, metric_fn)

    for layer_scores in scores.values():
        assert not torch.isnan(layer_scores).any(), "scores should not contain NaN"
        assert not torch.isinf(layer_scores).any(), "scores should not contain inf"
