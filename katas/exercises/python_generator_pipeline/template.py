"""
Python Generator Pipeline Kata

Learn how to chain generators together for memory-efficient data processing.

Your task:
Implement a generator pipeline that applies multiple transformations in sequence.
Each stage is a generator that takes an iterator and yields transformed values.

Key concepts:
- Generators can take iterators as input
- Chain generators by passing one to another
- Pipeline processes items one at a time (lazy evaluation)
- No intermediate lists are created (memory efficient)
"""

from typing import Iterator


def generator_pipeline(numbers: Iterator[int]) -> Iterator[int]:
    """
    Generator pipeline that chains multiple transformations.

    Pipeline stages:
    1. Square each number (n -> n²)
    2. Filter for even numbers
    3. Double the result (n -> 2n)

    Args:
        numbers: Input numbers

    Yields:
        Transformed numbers

    Example:
        >>> list(generator_pipeline(range(5)))
        [0, 8, 32]

        Explanation:
        - 0: 0² = 0 (even) → 0 × 2 = 0
        - 1: 1² = 1 (odd) → filtered out
        - 2: 2² = 4 (even) → 4 × 2 = 8
        - 3: 3² = 9 (odd) → filtered out
        - 4: 4² = 16 (even) → 16 × 2 = 32

    Hints:
        - Define three inner generator functions (one for each stage)
        - Each generator takes an iterator and yields transformed values
        - Chain them: double_gen(filter_even(square_gen(numbers)))
        - Return the final generator (don't consume it with list())
    """
    # TODO: Implement generator pipeline
    #
    # Define three generators:
    #
    # def square_gen(nums):
    #     for n in nums:
    #         yield n ** 2
    #
    # def filter_even(nums):
    #     for n in nums:
    #         if n % 2 == 0:
    #             yield n
    #
    # def double_gen(nums):
    #     for n in nums:
    #         yield n * 2
    #
    # Then chain and return them
    pass
