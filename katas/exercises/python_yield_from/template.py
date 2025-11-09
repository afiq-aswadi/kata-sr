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

    Hints:
        - Iterate through items in the list
        - Check if item is a list: isinstance(item, list)
        - If it's a list: yield from flatten_nested(item)
          - This delegates to recursive call
          - All values from sub-generator are yielded
        - If it's not a list: yield item
        - This is recursive - base case is non-list items
    """
    # TODO: Implement recursive flattening with yield from
    #
    # 1. for item in nested:
    # 2.     if isinstance(item, list):
    # 3.         yield from flatten_nested(item)  # Recursive delegation
    # 4.     else:
    # 5.         yield item  # Base case
    pass
