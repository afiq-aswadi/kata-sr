"""
Tests for Retry Decorator Kata
"""

import pytest
import time


def test_retry_succeeds_on_first_attempt(solution):
    """Test that retry decorator doesn't interfere when function succeeds."""
    retry = solution.retry

    call_count = 0

    @retry(max_attempts=3, backoff=0.01)
    def always_succeeds():
        nonlocal call_count
        call_count += 1
        return "success"

    result = always_succeeds()
    assert result == "success"
    assert call_count == 1


def test_retry_retries_on_failure(solution):
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


def test_retry_raises_after_max_attempts(solution):
    """Test that retry decorator raises exception after max attempts."""
    retry = solution.retry

    attempt_count = 0

    @retry(max_attempts=3, backoff=0.01)
    def always_fails():
        nonlocal attempt_count
        attempt_count += 1
        raise ValueError("Always fails!")

    with pytest.raises(ValueError, match="Always fails!"):
        always_fails()

    assert attempt_count == 3


def test_retry_exponential_backoff(solution):
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

    assert 0.04 < delay1 < 0.08, f"First delay {delay1}s outside expected range ~0.05s"
    assert 0.08 < delay2 < 0.15, f"Second delay {delay2}s outside expected range ~0.10s"


def test_retry_preserves_metadata(solution):
    """Test that retry decorator preserves function metadata."""
    retry = solution.retry

    @retry(max_attempts=2, backoff=0.01)
    def example_function():
        """Example docstring."""
        return "ok"

    assert example_function.__name__ == "example_function"
    assert example_function.__doc__ == "Example docstring."


def test_retry_with_different_max_attempts(solution):
    """Test retry with different max_attempts values."""
    retry = solution.retry

    attempt_count = 0

    @retry(max_attempts=5, backoff=0.01)
    def needs_five_tries():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 5:
            raise ValueError("Not yet!")
        return "success"

    result = needs_five_tries()
    assert result == "success"
    assert attempt_count == 5


def test_retry_with_different_backoff(solution):
    """Test retry with different backoff values."""
    retry = solution.retry

    attempt_times = []

    @retry(max_attempts=2, backoff=0.1)
    def custom_backoff():
        attempt_times.append(time.perf_counter())
        if len(attempt_times) < 2:
            raise ValueError("Retry!")
        return "done"

    custom_backoff()

    # With backoff=0.1, first delay should be 0.1 * (2^0) = 0.1s
    delay = attempt_times[1] - attempt_times[0]
    assert 0.08 < delay < 0.15, f"Delay {delay}s outside expected range ~0.1s"


def test_retry_with_arguments(solution):
    """Test that retry works with function arguments."""
    retry = solution.retry

    attempt_count = 0

    @retry(max_attempts=3, backoff=0.01)
    def add_with_retry(a, b):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 2:
            raise ValueError("Not yet!")
        return a + b

    result = add_with_retry(2, 3)
    assert result == 5
    assert attempt_count == 2


def test_retry_with_kwargs(solution):
    """Test that retry works with keyword arguments."""
    retry = solution.retry

    attempt_count = 0

    @retry(max_attempts=3, backoff=0.01)
    def greet_with_retry(name, greeting="Hello"):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 2:
            raise ValueError("Not yet!")
        return f"{greeting}, {name}!"

    result = greet_with_retry("Alice", greeting="Hi")
    assert result == "Hi, Alice!"


def test_retry_with_methods(solution):
    """Test that retry works with class methods."""
    retry = solution.retry

    class APIClient:
        def __init__(self):
            self.attempt_count = 0

        @retry(max_attempts=3, backoff=0.01)
        def fetch_data(self):
            self.attempt_count += 1
            if self.attempt_count < 2:
                raise ConnectionError("Network error")
            return {"data": "success"}

    client = APIClient()
    result = client.fetch_data()
    assert result == {"data": "success"}
    assert client.attempt_count == 2


def test_retry_different_exception_types(solution):
    """Test that retry handles different exception types."""
    retry = solution.retry

    attempt_count = 0

    @retry(max_attempts=3, backoff=0.01)
    def raises_different_errors():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise ConnectionError("Network error")
        elif attempt_count == 2:
            raise TimeoutError("Timeout")
        return "success"

    result = raises_different_errors()
    assert result == "success"
    assert attempt_count == 3


def test_retry_no_delay_on_last_attempt(solution):
    """Test that no delay happens after the last failed attempt."""
    retry = solution.retry

    attempt_times = []

    @retry(max_attempts=3, backoff=0.05)
    def always_fails():
        attempt_times.append(time.perf_counter())
        raise ValueError("Fail")

    start_time = time.perf_counter()
    with pytest.raises(ValueError):
        always_fails()
    total_time = time.perf_counter() - start_time

    # Should have delays after 1st and 2nd attempts only
    # Total delay: 0.05 * (2^0) + 0.05 * (2^1) = 0.05 + 0.10 = 0.15s
    # Allow some margin for execution time
    assert total_time < 0.25, "Should not delay after final attempt"
