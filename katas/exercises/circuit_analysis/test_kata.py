"""
Tests for Circuit Analysis kata
"""

import pytest
import torch

try:
    from template import analyze_circuit
except ImportError:
    from reference import analyze_circuit


class TestCircuitAnalysis:
    """Test circuit analysis function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        results = torch.tensor([
            [0.01, 0.15, 0.03],
            [0.05, 0.45, 0.08]
        ])

        analysis = analyze_circuit(results)

        assert isinstance(analysis, dict)

    def test_has_required_keys(self):
        """Test that result has required keys."""
        results = torch.zeros(3, 4)

        analysis = analyze_circuit(results)

        assert "important_heads" in analysis
        assert "max_effect" in analysis
        assert "max_head" in analysis

    def test_identifies_important_heads(self):
        """Test that important heads are correctly identified."""
        results = torch.tensor([
            [0.01, 0.02, 0.15, 0.03],
            [0.05, 0.45, 0.08, 0.12],
            [0.02, 0.03, 0.01, 0.02]
        ])

        analysis = analyze_circuit(results, threshold=0.1)

        important = analysis["important_heads"]

        # Should find heads with values > 0.1
        assert (0, 2) in important  # 0.15
        assert (1, 1) in important  # 0.45
        assert (1, 3) in important  # 0.12

    def test_finds_maximum(self):
        """Test that maximum effect is correctly found."""
        results = torch.tensor([
            [0.01, 0.15, 0.03],
            [0.05, 0.45, 0.08]
        ])

        analysis = analyze_circuit(results)

        assert analysis["max_effect"] == pytest.approx(0.45)
        assert analysis["max_head"] == (1, 1)

    def test_empty_important_heads(self):
        """Test with no heads above threshold."""
        results = torch.tensor([
            [0.01, 0.02, 0.03],
            [0.04, 0.05, 0.06]
        ])

        analysis = analyze_circuit(results, threshold=0.5)

        assert len(analysis["important_heads"]) == 0

    def test_all_heads_important(self):
        """Test when all heads exceed threshold."""
        results = torch.ones(2, 3)

        analysis = analyze_circuit(results, threshold=0.5)

        assert len(analysis["important_heads"]) == 6  # All 2x3 heads

    def test_max_head_is_tuple(self):
        """Test that max_head is a tuple of two integers."""
        results = torch.randn(3, 4)

        analysis = analyze_circuit(results)

        assert isinstance(analysis["max_head"], tuple)
        assert len(analysis["max_head"]) == 2
        assert isinstance(analysis["max_head"][0], int)
        assert isinstance(analysis["max_head"][1], int)

    def test_important_heads_are_tuples(self):
        """Test that important_heads contains tuples."""
        results = torch.tensor([[0.5, 0.2], [0.3, 0.6]])

        analysis = analyze_circuit(results, threshold=0.1)

        for head in analysis["important_heads"]:
            assert isinstance(head, tuple)
            assert len(head) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
