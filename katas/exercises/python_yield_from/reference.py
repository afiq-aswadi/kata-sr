"""
Reference implementation for Python Yield From kata
"""

from typing import Iterator, List, Any


def flatten_nested(nested: List[Any]) -> Iterator[Any]:
    """
    Recursively flatten a nested list structure using yield from.

    yield from delegates to sub-generators, properly handling
    recursive structures.

    Args:
        nested: Nested list structure

    Yields:
        Flattened elements

    Example:
        >>> list(flatten_nested([1, [2, 3], [4, [5, 6]]]))
        [1, 2, 3, 4, 5, 6]
    """
    for item in nested:
        if isinstance(item, list):
            # Delegate to recursive call using yield from
            yield from flatten_nested(item)
        else:
            yield item
