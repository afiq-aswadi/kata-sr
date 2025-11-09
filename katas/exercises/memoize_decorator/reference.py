"""
Memoization Decorator Kata - Reference Implementation
"""

import functools
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')


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
