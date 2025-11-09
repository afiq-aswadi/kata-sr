"""
Tests for Timing Decorator Kata
"""

import pytest
import time
import re
from io import StringIO
import sys


def test_timing_decorator_measures_execution_time(solution):
    """Test that timing decorator correctly measures execution time."""
    timing = solution.timing

    # Capture stdout
    captured_output = StringIO()
    sys.stdout = captured_output

    @timing
    def slow_function():
        time.sleep(0.1)
        return "done"

    result = slow_function()

    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()

    # Check function still works
    assert result == "done"

    # Check timing message was printed
    assert "slow_function" in output
    assert "took" in output
    assert "s" in output

    # Extract and verify timing
    match = re.search(r"(\d+\.\d+)s", output)
    assert match is not None, "No timing found in output"
    elapsed = float(match.group(1))
    assert 0.09 < elapsed < 0.15, f"Timing {elapsed}s outside expected range"


def test_timing_decorator_preserves_metadata(solution):
    """Test that timing decorator preserves function metadata."""
    timing = solution.timing

    @timing
    def example_function():
        """Example docstring."""
        pass

    assert example_function.__name__ == "example_function"
    assert example_function.__doc__ == "Example docstring."


def test_timing_decorator_with_arguments(solution):
    """Test that timing decorator works with function arguments."""
    timing = solution.timing

    captured_output = StringIO()
    sys.stdout = captured_output

    @timing
    def add(a, b):
        time.sleep(0.01)
        return a + b

    result = add(2, 3)

    sys.stdout = sys.__stdout__

    assert result == 5


def test_timing_decorator_with_kwargs(solution):
    """Test that timing decorator works with keyword arguments."""
    timing = solution.timing

    captured_output = StringIO()
    sys.stdout = captured_output

    @timing
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"

    result = greet("Alice", greeting="Hi")

    sys.stdout = sys.__stdout__

    assert result == "Hi, Alice!"


def test_timing_decorator_with_methods(solution):
    """Test that timing decorator works with class methods."""
    timing = solution.timing

    captured_output = StringIO()
    sys.stdout = captured_output

    class Calculator:
        @timing
        def multiply(self, a, b):
            return a * b

    calc = Calculator()
    result = calc.multiply(4, 5)

    sys.stdout = sys.__stdout__

    assert result == 20


def test_timing_decorator_handles_exceptions(solution):
    """Test that timing decorator allows exceptions to propagate."""
    timing = solution.timing

    captured_output = StringIO()
    sys.stdout = captured_output

    @timing
    def failing_function():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        failing_function()

    sys.stdout = sys.__stdout__

    # Should still print timing before exception propagates
    output = captured_output.getvalue()
    assert "failing_function" in output


def test_timing_format(solution):
    """Test that timing output format matches specification."""
    timing = solution.timing

    captured_output = StringIO()
    sys.stdout = captured_output

    @timing
    def fast_function():
        time.sleep(0.05)
        return 42

    fast_function()

    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()

    # Check format: Function '{name}' took {time:.4f}s
    assert re.match(r"Function 'fast_function' took \d+\.\d{4}s", output)
