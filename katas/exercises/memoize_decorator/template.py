"""
Memoization Decorator Kata

Implement a decorator that caches function results to avoid redundant computation.
"""

import functools
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')


def memoize(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator that caches function results based on arguments.

    The cache should store results keyed by the function's arguments.
    If called again with the same arguments, return the cached result
    instead of recomputing.

    Example:
        @memoize
        def fibonacci(n):
            if n < 2:
                return n
            return fibonacci(n-1) + fibonacci(n-2)

        fibonacci(10)  # Computes once
        fibonacci(10)  # Returns cached result instantly
    """
    # BLANK_START
    raise NotImplementedError(
        "Create a cache dict, wrapper with @functools.wraps, "
        "check cache using (args, tuple(sorted(kwargs.items()))) as key"
    )
    # BLANK_END
