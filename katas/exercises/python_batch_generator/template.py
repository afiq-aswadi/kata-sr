"""
Python Batch Generator Kata

Learn the basics of generators and the yield keyword.

Your task:
Implement a generator that yields batches from a sequence. This demonstrates
the fundamental concept of generators: producing values one at a time while
maintaining state between calls.

Key concepts:
- yield produces a value and suspends the function
- Generator state is preserved between yields
- Generators are memory efficient (don't materialize entire sequence)
"""

from typing import Iterator, List, Any


def batch_generator(data: Iterator[Any], batch_size: int) -> Iterator[List[Any]]:
    """
    Generator that yields batches from a sequence.

    This is the most basic generator pattern - accumulate items and yield
    when batch is full.

    Args:
        data: Iterable to batch
        batch_size: Size of each batch

    Yields:
        Lists of size batch_size (last batch may be smaller)

    Example:
        >>> list(batch_generator(range(5), 2))
        [[0, 1], [2, 3], [4]]

        >>> list(batch_generator([1, 2, 3, 4, 5, 6], 3))
        [[1, 2, 3], [4, 5, 6]]

    """
    # BLANK_START
    raise NotImplementedError("Create a batch list, iterate through data, yield when full, don't forget final batch")
    # BLANK_END
