"""
Reference implementation for Python Batch Generator kata
"""

from typing import Iterator, List, Any


def batch_generator(data: Iterator[Any], batch_size: int) -> Iterator[List[Any]]:
    """
    Generator that yields batches from a sequence.

    Args:
        data: Iterable to batch
        batch_size: Size of each batch

    Yields:
        Lists of size batch_size (last batch may be smaller)

    Example:
        >>> list(batch_generator(range(5), 2))
        [[0, 1], [2, 3], [4]]
    """
    batch = []
    for item in data:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    # Yield remaining items if any
    if batch:
        yield batch
