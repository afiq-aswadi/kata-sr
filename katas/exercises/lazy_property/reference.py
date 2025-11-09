"""
Lazy Property Kata - Reference Implementation
"""

import functools
from typing import Any, Callable, TypeVar

T = TypeVar('T')


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
