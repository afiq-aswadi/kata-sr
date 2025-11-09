"""Tests for Fibonacci DP kata."""

try:
    from user_kata import fibonacci_dp
except ModuleNotFoundError:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).with_name("reference.py")
    spec = importlib.util.spec_from_file_location("reference", module_path)
    reference = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(reference)
    fibonacci_dp = reference.fibonacci_dp  # type: ignore


def test_base_cases():
    """Test base cases."""
    assert fibonacci_dp(0) == 0, "fib(0) should be 0"
    assert fibonacci_dp(1) == 1, "fib(1) should be 1"


def test_small_values():
    """Test small Fibonacci numbers."""
    expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    for i, exp in enumerate(expected):
        assert fibonacci_dp(i) == exp, f"fib({i}) should be {exp}"


def test_larger_values():
    """Test larger Fibonacci numbers."""
    assert fibonacci_dp(10) == 55
    assert fibonacci_dp(15) == 610
    assert fibonacci_dp(20) == 6765


def test_very_large_value():
    """Test that memoization allows computing large values efficiently."""
    result = fibonacci_dp(100)
    assert result == 354224848179261915075, "fib(100) should be correct"


def test_repeated_calls():
    """Repeated calls should return same result."""
    n = 25
    result1 = fibonacci_dp(n)
    result2 = fibonacci_dp(n)
    assert result1 == result2, "repeated calls should give same result"
