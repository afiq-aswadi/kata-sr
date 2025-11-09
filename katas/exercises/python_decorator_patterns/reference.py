"""
Python Decorator Patterns Kata - Reference Implementation
"""

import time
import functools
from typing import Any, Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')


def timing(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator that measures and prints the execution time of a function.
    """
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"Function '{func.__name__}' took {elapsed:.4f}s")
        return result
    return wrapper


def memoize(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator that caches function results based on arguments.
    """
    cache: dict = {}

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # Create a hashable cache key from args and kwargs
        key = (args, tuple(sorted(kwargs.items())))

        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    return wrapper


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


class lazy_property:
    """
    Descriptor-based decorator for lazy evaluation of properties.
    """

    def __init__(self, func: Callable[[Any], T]) -> None:
        self.func = func
        functools.update_wrapper(self, func)

    def __get__(self, obj: Any, objtype: Any = None) -> T:
        if obj is None:
            return self

        # Use the function's name as the cache key
        attr_name = self.func.__name__

        # Check if value is already cached
        if attr_name not in obj.__dict__:
            # Compute and cache the value
            obj.__dict__[attr_name] = self.func(obj)

        return obj.__dict__[attr_name]
