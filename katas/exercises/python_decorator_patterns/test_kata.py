"""
Tests for Python Decorator Patterns Kata
"""

import pytest
import time
from io import StringIO
import sys


def test_timing_decorator_measures_execution_time(solution):
    """Test that timing decorator correctly measures execution time."""
    timing = solution.timing

    # Capture stdout to check the printed message
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

    # Check timing is approximately correct (within tolerance)
    # Extract the time value from output
    import re
    match = re.search(r"(\d+\.\d+)s", output)
    assert match is not None
    elapsed = float(match.group(1))
    assert 0.09 < elapsed < 0.15  # Allow some tolerance


def test_timing_decorator_preserves_metadata(solution):
    """Test that timing decorator preserves function metadata."""
    timing = solution.timing

    @timing
    def example_function():
        """Example docstring."""
        pass

    assert example_function.__name__ == "example_function"
    assert example_function.__doc__ == "Example docstring."


def test_memoize_decorator_caches_results(solution):
    """Test that memoization decorator caches results."""
    memoize = solution.memoize

    call_count = 0

    @memoize
    def expensive_function(n):
        nonlocal call_count
        call_count += 1
        time.sleep(0.01)
        return n * 2

    # First call - should execute
    result1 = expensive_function(5)
    assert result1 == 10
    assert call_count == 1

    # Second call with same args - should use cache
    result2 = expensive_function(5)
    assert result2 == 10
    assert call_count == 1  # Should not increment

    # Call with different args - should execute
    result3 = expensive_function(10)
    assert result3 == 20
    assert call_count == 2


def test_memoize_decorator_handles_multiple_arguments(solution):
    """Test that memoization works with multiple arguments."""
    memoize = solution.memoize

    call_count = 0

    @memoize
    def add(a, b):
        nonlocal call_count
        call_count += 1
        return a + b

    assert add(1, 2) == 3
    assert call_count == 1

    assert add(1, 2) == 3
    assert call_count == 1  # Cached

    assert add(2, 3) == 5
    assert call_count == 2  # New args


def test_memoize_decorator_handles_kwargs(solution):
    """Test that memoization works with keyword arguments."""
    memoize = solution.memoize

    call_count = 0

    @memoize
    def greet(name, greeting="Hello"):
        nonlocal call_count
        call_count += 1
        return f"{greeting}, {name}!"

    assert greet("Alice") == "Hello, Alice!"
    assert call_count == 1

    assert greet("Alice") == "Hello, Alice!"
    assert call_count == 1  # Cached

    assert greet("Alice", greeting="Hi") == "Hi, Alice!"
    assert call_count == 2  # Different kwargs


def test_retry_decorator_retries_on_failure(solution):
    """Test that retry decorator retries failed functions."""
    retry = solution.retry

    attempt_count = 0

    @retry(max_attempts=3, backoff=0.01)
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError("Not yet!")
        return "success"

    result = flaky_function()
    assert result == "success"
    assert attempt_count == 3


def test_retry_decorator_raises_after_max_attempts(solution):
    """Test that retry decorator raises exception after max attempts."""
    retry = solution.retry

    @retry(max_attempts=3, backoff=0.01)
    def always_fails():
        raise ValueError("Always fails!")

    with pytest.raises(ValueError, match="Always fails!"):
        always_fails()


def test_retry_decorator_exponential_backoff(solution):
    """Test that retry decorator uses exponential backoff."""
    retry = solution.retry

    attempt_times = []

    @retry(max_attempts=3, backoff=0.05)
    def track_attempts():
        attempt_times.append(time.perf_counter())
        if len(attempt_times) < 3:
            raise ValueError("Retry!")
        return "done"

    track_attempts()

    # Check delays are approximately correct
    # First delay: 0.05 * (2^0) = 0.05s
    # Second delay: 0.05 * (2^1) = 0.10s
    delay1 = attempt_times[1] - attempt_times[0]
    delay2 = attempt_times[2] - attempt_times[1]

    assert 0.04 < delay1 < 0.08  # ~0.05s with tolerance
    assert 0.08 < delay2 < 0.15  # ~0.10s with tolerance


def test_retry_decorator_preserves_metadata(solution):
    """Test that retry decorator preserves function metadata."""
    retry = solution.retry

    @retry(max_attempts=2, backoff=0.01)
    def example_function():
        """Example docstring."""
        return "ok"

    assert example_function.__name__ == "example_function"
    assert example_function.__doc__ == "Example docstring."


def test_lazy_property_evaluates_once(solution):
    """Test that lazy_property only evaluates once."""
    lazy_property = solution.lazy_property

    eval_count = 0

    class MyClass:
        @lazy_property
        def expensive_property(self):
            nonlocal eval_count
            eval_count += 1
            time.sleep(0.01)
            return "computed"

    obj = MyClass()

    # First access - should compute
    result1 = obj.expensive_property
    assert result1 == "computed"
    assert eval_count == 1

    # Second access - should use cached value
    result2 = obj.expensive_property
    assert result2 == "computed"
    assert eval_count == 1  # Should not increment


def test_lazy_property_separate_instances(solution):
    """Test that lazy_property maintains separate cache per instance."""
    lazy_property = solution.lazy_property

    class Counter:
        def __init__(self, value):
            self.value = value

        @lazy_property
        def doubled(self):
            return self.value * 2

    obj1 = Counter(5)
    obj2 = Counter(10)

    assert obj1.doubled == 10
    assert obj2.doubled == 20
    assert obj1.doubled == 10  # Still cached separately


def test_decorators_work_with_methods(solution):
    """Test that decorators work with class methods."""
    timing = solution.timing
    memoize = solution.memoize

    class Calculator:
        @memoize
        def fibonacci(self, n):
            if n < 2:
                return n
            return self.fibonacci(n - 1) + self.fibonacci(n - 2)

        @timing
        def slow_method(self):
            time.sleep(0.01)
            return "done"

    # Suppress timing output
    captured_output = StringIO()
    sys.stdout = captured_output

    calc = Calculator()
    assert calc.fibonacci(10) == 55
    result = calc.slow_method()

    sys.stdout = sys.__stdout__

    assert result == "done"


def test_edge_case_no_arguments(solution):
    """Test decorators with functions that take no arguments."""
    memoize = solution.memoize

    call_count = 0

    @memoize
    def no_args():
        nonlocal call_count
        call_count += 1
        return 42

    assert no_args() == 42
    assert call_count == 1

    assert no_args() == 42
    assert call_count == 1  # Cached
