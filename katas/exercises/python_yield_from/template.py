"""
Python Yield From Kata

Learn the yield from syntax for delegating to sub-generators.

Your task:
Implement a recursive generator that flattens nested lists using yield from.
This demonstrates how yield from completely delegates iteration to a sub-generator.

Key concepts:
- yield from passes all values from sub-generator to caller
- More efficient than: for item in sub_gen: yield item
- Essential for recursive generators
- Transparently handles send() and throw() to sub-generators
"""

from typing import Iterator, List, Any


def flatten_nested(nested: List[Any]) -> Iterator[Any]:
    """
    Recursively flatten a nested list structure using yield from.

    yield from delegates to sub-generators, properly handling
    recursive structures.

    Args:
        nested: Nested list structure (can contain lists or non-list items)

    Yields:
        Flattened elements in order

    Example:
        >>> list(flatten_nested([1, [2, 3], [4, [5, 6]]]))
        [1, 2, 3, 4, 5, 6]

        >>> list(flatten_nested([1, [2, [3, [4, [5]]]]]))
        [1, 2, 3, 4, 5]

    """
    # BLANK_START
    raise NotImplementedError("Iterate through items, use 'yield from flatten_nested(item)' for lists, 'yield item' otherwise")
    # BLANK_END
