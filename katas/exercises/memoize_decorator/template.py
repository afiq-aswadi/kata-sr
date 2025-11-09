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
    # TODO: Implement the memoization decorator
    # Hints:
    # 1. Create a cache dictionary outside the wrapper (closure variable)
    # 2. Convert args and kwargs into a hashable cache key
    #    - args is already a tuple (hashable)
    #    - kwargs needs to be converted: tuple(sorted(kwargs.items()))
    # 3. Check if key exists in cache before calling function
    # 4. Store result in cache if not present
    # 5. Return the cached or computed result
    # 6. Don't forget @functools.wraps(func)
    pass
