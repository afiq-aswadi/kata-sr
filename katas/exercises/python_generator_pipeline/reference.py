"""
Reference implementation for Python Generator Pipeline kata
"""

from typing import Iterator


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
