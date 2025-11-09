"""
Timing Decorator Kata - Reference Implementation
"""

import time
import functools
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')


def timing(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator that measures and prints the execution time of a function.
    """
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"Function '{func.__name__}' took {elapsed:.4f}s")
    return wrapper
