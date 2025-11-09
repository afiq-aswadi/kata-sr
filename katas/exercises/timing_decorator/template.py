"""
Timing Decorator Kata

Implement a decorator that measures and reports function execution time.
"""

import time
import functools
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')


def timing(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator that measures and prints the execution time of a function.

    Should print: "Function '{func_name}' took {elapsed:.4f}s"

    Example:
        @timing
        def slow_function():
            time.sleep(0.1)
            return "done"

        slow_function()  # Prints: Function 'slow_function' took 0.1002s
    """
    # BLANK_START
    raise NotImplementedError(
        "Create a wrapper function using @functools.wraps(func), "
        "measure time with time.perf_counter(), and print elapsed time"
    )
    # BLANK_END
