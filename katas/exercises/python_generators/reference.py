"""
Reference implementation for Python Generators kata

This module demonstrates proper use of:
- yield for basic generators
- Generator pipelines
- send() for bidirectional communication
- try/finally for cleanup
- yield from for delegation
"""

from typing import Iterator, List, Any, Union


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


def generator_pipeline(numbers: Iterator[int]) -> Iterator[int]:
    """
    Generator pipeline that chains multiple transformations.

    Pipeline:
    1. Square each number
    2. Filter for even numbers
    3. Double the result

    Args:
        numbers: Input numbers

    Yields:
        Transformed numbers

    Example:
        >>> list(generator_pipeline(range(5)))
        [0, 8, 32]
    """
    # Stage 1: Square
    def square_gen(nums):
        for n in nums:
            yield n ** 2

    # Stage 2: Filter even
    def filter_even(nums):
        for n in nums:
            if n % 2 == 0:
                yield n

    # Stage 3: Double
    def double_gen(nums):
        for n in nums:
            yield n * 2

    # Chain the pipeline
    return double_gen(filter_even(square_gen(numbers)))


def running_average() -> Iterator[float]:
    """
    Stateful generator that computes running average using send().

    This generator maintains state across yields and receives values
    via send() to compute a running average.

    Yields:
        Current running average after each value is sent

    Example:
        >>> avg = running_average()
        >>> next(avg)  # Prime the generator
        >>> avg.send(10)
        10.0
        >>> avg.send(20)
        15.0
        >>> avg.send(30)
        20.0
    """
    total = 0
    count = 0
    average = None

    while True:
        # Receive value via send()
        value = yield average
        total += value
        count += 1
        average = total / count


def file_reader_generator(filepath: str) -> Iterator[str]:
    """
    Generator that reads a file line by line with proper cleanup.

    Uses try/finally to ensure file is closed even if generator
    is not fully consumed or an exception occurs.

    Args:
        filepath: Path to file to read

    Yields:
        Lines from the file

    Example:
        >>> gen = file_reader_generator("data.txt")
        >>> first_line = next(gen)
        >>> gen.close()  # Cleanup happens here
    """
    f = open(filepath, 'r')
    try:
        for line in f:
            yield line
    finally:
        # Cleanup code - always executes
        f.close()


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
