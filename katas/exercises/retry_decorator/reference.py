"""
Retry Decorator Kata - Reference Implementation
"""

import time
import functools
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')


def retry(max_attempts: int = 3, backoff: float = 1.0) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Parameterized decorator that retries a function on exception with exponential backoff.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = backoff * (2 ** attempt)
                        time.sleep(delay)

            # All attempts failed, re-raise the last exception
            raise last_exception

        return wrapper
    return decorator
