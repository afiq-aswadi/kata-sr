"""
Tests for Memoization Decorator Kata
"""

import pytest
import time


def test_memoize_caches_results(solution):
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


def test_memoize_handles_multiple_arguments(solution):
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


def test_memoize_handles_kwargs(solution):
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

    assert greet("Alice", greeting="Hi") == "Hi, Alice!"
    assert call_count == 2  # Cached with kwargs


def test_memoize_preserves_metadata(solution):
    """Test that memoize decorator preserves function metadata."""
    memoize = solution.memoize

    @memoize
    def example_function():
        """Example docstring."""
        return 42

    assert example_function.__name__ == "example_function"
    assert example_function.__doc__ == "Example docstring."


def test_memoize_no_arguments(solution):
    """Test memoization with functions that take no arguments."""
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


def test_memoize_with_fibonacci(solution):
    """Test memoization with recursive fibonacci (classic use case)."""
    memoize = solution.memoize

    call_count = 0

    @memoize
    def fibonacci(n):
        nonlocal call_count
        call_count += 1
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    result = fibonacci(10)
    assert result == 55

    # With memoization, should call fibonacci(10) through fibonacci(0) only once each
    # That's 11 calls total (0 through 10)
    assert call_count == 11

    # Calling again should not increase count
    fibonacci(10)
    assert call_count == 11


def test_memoize_with_methods(solution):
    """Test that memoization works with class methods."""
    memoize = solution.memoize

    class Calculator:
        def __init__(self):
            self.call_count = 0

        @memoize
        def square(self, n):
            self.call_count += 1
            return n * n

    calc = Calculator()
    assert calc.square(5) == 25
    assert calc.call_count == 1

    assert calc.square(5) == 25
    assert calc.call_count == 1  # Cached


def test_memoize_different_types(solution):
    """Test memoization with different argument types."""
    memoize = solution.memoize

    call_count = 0

    @memoize
    def process(value):
        nonlocal call_count
        call_count += 1
        return str(value)

    # Test with different types
    assert process(42) == "42"
    assert call_count == 1

    assert process(42) == "42"
    assert call_count == 1  # Cached

    assert process("hello") == "hello"
    assert call_count == 2

    assert process("hello") == "hello"
    assert call_count == 2  # Cached
