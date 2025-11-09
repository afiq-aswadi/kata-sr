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
    # TODO: Implement the timing decorator
    # Hints:
    # 1. Use @functools.wraps(func) to preserve function metadata
    # 2. Create a wrapper function that will replace the original
    # 3. Use time.perf_counter() before and after calling func
    # 4. Calculate elapsed time and print the message
    # 5. Return the function's result
    # 6. Return the wrapper function
    pass
