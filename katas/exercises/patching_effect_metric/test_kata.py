"""
Tests for Patching Effect Metric kata
"""

import pytest
import torch

try:
    from template import compute_patching_effect
except ImportError:
    from reference import compute_patching_effect


class TestPatchingEffectMetric:
    """Test normalized patching effect computation."""

    def test_full_restoration_gives_one(self):
        """Test that fully restoring clean behavior gives effect = 1.0."""
        clean_logits = torch.zeros(1, 5, 100)
        clean_logits[0, -1, 50] = 5.0  # answer
        clean_logits[0, -1, 75] = 2.0  # wrong

        corrupt_logits = torch.zeros(1, 5, 100)
        corrupt_logits[0, -1, 50] = 2.0  # answer
        corrupt_logits[0, -1, 75] = 4.0  # wrong

        # Patched = Clean (full restoration)
        patched_logits = clean_logits.clone()

        effect = compute_patching_effect(
            patched_logits, clean_logits, corrupt_logits, 50, 75
        )

        assert effect == pytest.approx(1.0, abs=0.01)

    def test_no_effect_gives_zero(self):
        """Test that no change from corrupt gives effect = 0.0."""
        clean_logits = torch.zeros(1, 5, 100)
        clean_logits[0, -1, 50] = 5.0
        clean_logits[0, -1, 75] = 2.0

        corrupt_logits = torch.zeros(1, 5, 100)
        corrupt_logits[0, -1, 50] = 2.0
        corrupt_logits[0, -1, 75] = 4.0

        # Patched = Corrupt (no effect)
        patched_logits = corrupt_logits.clone()

        effect = compute_patching_effect(
            patched_logits, clean_logits, corrupt_logits, 50, 75
        )

        assert effect == pytest.approx(0.0, abs=0.01)

    def test_partial_restoration(self):
        """Test partial restoration gives effect between 0 and 1."""
        clean_logits = torch.zeros(1, 5, 100)
        clean_logits[0, -1, 50] = 6.0  # clean diff = 6 - 2 = 4
        clean_logits[0, -1, 75] = 2.0

        corrupt_logits = torch.zeros(1, 5, 100)
        corrupt_logits[0, -1, 50] = 2.0  # corrupt diff = 2 - 4 = -2
        corrupt_logits[0, -1, 75] = 4.0

        # Halfway restoration
        patched_logits = torch.zeros(1, 5, 100)
        patched_logits[0, -1, 50] = 4.0  # patched diff = 4 - 3 = 1
        patched_logits[0, -1, 75] = 3.0

        effect = compute_patching_effect(
            patched_logits, clean_logits, corrupt_logits, 50, 75
        )

        # Effect should be 0.5: (1 - (-2)) / (4 - (-2)) = 3/6 = 0.5
        assert 0.4 < effect < 0.6

    def test_overcorrection_gives_greater_than_one(self):
        """Test that overcorrecting gives effect > 1.0."""
        clean_logits = torch.zeros(1, 5, 100)
        clean_logits[0, -1, 50] = 5.0
        clean_logits[0, -1, 75] = 2.0

        corrupt_logits = torch.zeros(1, 5, 100)
        corrupt_logits[0, -1, 50] = 2.0
        corrupt_logits[0, -1, 75] = 4.0

        # Overcorrected (better than clean)
        patched_logits = torch.zeros(1, 5, 100)
        patched_logits[0, -1, 50] = 8.0  # Even higher than clean
        patched_logits[0, -1, 75] = 1.0

        effect = compute_patching_effect(
            patched_logits, clean_logits, corrupt_logits, 50, 75
        )

        assert effect > 1.0

    def test_returns_float(self):
        """Test that result is a Python float."""
        clean_logits = torch.ones(1, 5, 100)
        corrupt_logits = torch.zeros(1, 5, 100)
        patched_logits = torch.ones(1, 5, 100) * 0.5

        effect = compute_patching_effect(
            patched_logits, clean_logits, corrupt_logits, 10, 20
        )

        assert isinstance(effect, float)

    def test_handles_equal_clean_corrupt(self):
        """Test graceful handling when clean and corrupt are identical."""
        logits = torch.zeros(1, 5, 100)
        logits[0, -1, 50] = 3.0
        logits[0, -1, 75] = 3.0

        # All same - denominator would be zero
        effect = compute_patching_effect(
            logits, logits, logits, 50, 75
        )

        # Should handle gracefully (return 0.0 or similar)
        assert isinstance(effect, float)
        assert abs(effect) < 10.0  # Should be reasonable

    def test_negative_effect_when_making_worse(self):
        """Test that making things worse gives negative effect."""
        clean_logits = torch.zeros(1, 5, 100)
        clean_logits[0, -1, 50] = 5.0
        clean_logits[0, -1, 75] = 2.0

        corrupt_logits = torch.zeros(1, 5, 100)
        corrupt_logits[0, -1, 50] = 4.0
        corrupt_logits[0, -1, 75] = 3.0

        # Make it even worse than corrupt
        patched_logits = torch.zeros(1, 5, 100)
        patched_logits[0, -1, 50] = 2.0
        patched_logits[0, -1, 75] = 5.0

        effect = compute_patching_effect(
            patched_logits, clean_logits, corrupt_logits, 50, 75
        )

        assert effect < 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
