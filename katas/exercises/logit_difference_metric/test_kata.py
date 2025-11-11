"""
Tests for Logit Difference Metric kata
"""

import pytest
import torch

try:
except ImportError:
    from reference import compute_logit_diff


try:
    from user_kata import compute_logit_diff
except ImportError:
    from .reference import compute_logit_diff


class TestLogitDifference:
    """Test logit difference computation."""

    def test_basic_difference(self):
        """Test basic logit difference calculation."""
        # Create simple logits
        logits = torch.zeros(1, 5, 100)
        logits[0, -1, 50] = 2.0  # answer token
        logits[0, -1, 75] = 1.0  # wrong token

        diff = compute_logit_diff(logits, answer_token=50, wrong_token=75)

        assert diff.item() == pytest.approx(1.0)

    def test_negative_difference(self):
        """Test when wrong token has higher logit."""
        logits = torch.zeros(1, 5, 100)
        logits[0, -1, 50] = 1.0  # answer token
        logits[0, -1, 75] = 3.0  # wrong token

        diff = compute_logit_diff(logits, answer_token=50, wrong_token=75)

        assert diff.item() == pytest.approx(-2.0)

    def test_position_argument(self):
        """Test different position indices."""
        logits = torch.zeros(1, 5, 100)
        # Set different values at different positions
        logits[0, 0, 10] = 5.0
        logits[0, 0, 20] = 2.0
        logits[0, 2, 10] = 3.0
        logits[0, 2, 20] = 1.0

        # Check first position
        diff_first = compute_logit_diff(logits, 10, 20, position=0)
        assert diff_first.item() == pytest.approx(3.0)

        # Check middle position
        diff_middle = compute_logit_diff(logits, 10, 20, position=2)
        assert diff_middle.item() == pytest.approx(2.0)

    def test_last_position_default(self):
        """Test that position=-1 uses last token."""
        logits = torch.zeros(1, 5, 100)
        logits[0, 4, 10] = 7.0  # last position
        logits[0, 4, 20] = 4.0

        diff = compute_logit_diff(logits, 10, 20)  # position=-1 by default

        assert diff.item() == pytest.approx(3.0)

    def test_zero_difference(self):
        """Test when logits are equal."""
        logits = torch.zeros(1, 5, 100)
        logits[0, -1, 50] = 2.5
        logits[0, -1, 75] = 2.5

        diff = compute_logit_diff(logits, 50, 75)

        assert diff.item() == pytest.approx(0.0)

    def test_returns_tensor(self):
        """Test that result is a tensor."""
        logits = torch.zeros(1, 5, 100)
        diff = compute_logit_diff(logits, 10, 20)

        assert isinstance(diff, torch.Tensor)
        assert diff.numel() == 1  # Scalar tensor

    def test_with_realistic_logits(self):
        """Test with realistic logit values."""
        logits = torch.randn(1, 10, 50257)  # GPT-2 vocab size
        logits[0, -1, 5000] = 10.5  # answer token
        logits[0, -1, 5001] = 8.3   # wrong token

        diff = compute_logit_diff(logits, 5000, 5001)

        assert diff.item() == pytest.approx(2.2, abs=0.01)

    def test_negative_position_indexing(self):
        """Test negative position indices."""
        logits = torch.zeros(1, 5, 100)
        logits[0, -2, 10] = 6.0  # second to last
        logits[0, -2, 20] = 4.0

        diff = compute_logit_diff(logits, 10, 20, position=-2)

        assert diff.item() == pytest.approx(2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
