"""
Retry Decorator Kata

Implement a parameterized decorator that retries failed functions with exponential backoff.
"""

import time
import functools
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')


def retry(max_attempts: int = 3, backoff: float = 1.0) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Parameterized decorator that retries a function on exception with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        backoff: Initial delay between retries in seconds (default: 1.0)

    The delay between retries follows exponential backoff:
        delay = backoff * (2 ** attempt_number)

    For example, with backoff=1.0:
        - After 1st failure: wait 1.0 * (2^0) = 1.0s
        - After 2nd failure: wait 1.0 * (2^1) = 2.0s
        - After 3rd failure: wait 1.0 * (2^2) = 4.0s

    If all attempts fail, re-raise the last exception.

    Example:
        @retry(max_attempts=3, backoff=0.5)
        def flaky_api_call():
            response = requests.get("https://api.example.com/data")
            return response.json()

        # Will retry up to 3 times with 0.5s, 1.0s, 2.0s delays
    """
    # TODO: Implement the retry decorator
    # Hints:
    # 1. This is a PARAMETERIZED decorator - it returns a decorator function
    # 2. You need THREE levels of nesting:
    #    - retry(params) -> returns decorator
    #    - decorator(func) -> returns wrapper
    #    - wrapper(*args, **kwargs) -> calls func with retries
    # 3. In the wrapper, use a for loop for attempts (0 to max_attempts-1)
    # 4. Wrap func() call in try/except
    # 5. On exception, calculate delay: backoff * (2 ** attempt)
    # 6. Use time.sleep(delay) before next retry
    # 7. If it's the last attempt, re-raise the exception
    # 8. Don't forget @functools.wraps(func) on the wrapper
    pass
