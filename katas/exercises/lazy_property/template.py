"""
Lazy Property Kata

Implement a descriptor-based decorator for lazy evaluation of properties.
"""

import functools
from typing import Any, Callable, TypeVar

T = TypeVar('T')


class lazy_property:
    """
    Descriptor-based decorator for lazy evaluation of properties.

    The property is computed once on first access and cached in the instance's
    __dict__. Subsequent accesses return the cached value without recomputation.

    Acts like @property but only evaluates once.

    Example:
        class DataProcessor:
            def __init__(self, data):
                self.data = data

            @lazy_property
            def processed_data(self):
                print("Processing...")  # Only prints once
                return expensive_computation(self.data)

        processor = DataProcessor([1, 2, 3])
        result1 = processor.processed_data  # Prints "Processing..."
        result2 = processor.processed_data  # Returns cached value
    """

    def __init__(self, func: Callable[[Any], T]) -> None:
        """
        Initialize the lazy property with the function to be lazily evaluated.

        Args:
            func: The method to be lazily evaluated
        """
        # TODO: Store the function
        # Hint: Use functools.update_wrapper(self, func) to copy metadata
        pass

    def __get__(self, obj: Any, objtype: Any = None) -> T:
        """
        Descriptor protocol: called when the attribute is accessed.

        Args:
            obj: The instance accessing the property (None if accessed via class)
            objtype: The class of the instance

        Returns:
            The computed or cached property value
        """
        # TODO: Implement the descriptor protocol
        # Hints:
        # 1. If obj is None (class access), return self
        # 2. Use self.func.__name__ to get the attribute name
        # 3. Check if the value is already cached in obj.__dict__
        # 4. If not cached:
        #    - Call self.func(obj) to compute the value
        #    - Store it in obj.__dict__[attr_name]
        # 5. Return obj.__dict__[attr_name]
        pass
