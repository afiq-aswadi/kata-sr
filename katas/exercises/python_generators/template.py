"""
Python Generators Kata - Template

Learn about Python generators and the iteration protocol.

Your tasks:
1. Implement batch_generator() - yields batches from a sequence
2. Implement generator_pipeline() - chains multiple generators
3. Implement running_average() - uses send() for bidirectional communication
4. Implement file_reader_generator() - ensures cleanup with try/finally
5. Implement flatten_nested() - uses yield from for recursive delegation

Key concepts to practice:
- yield vs return (yield produces values while maintaining state)
- Generator state suspension/resumption
- send() for passing values into generator
- try/finally for guaranteed cleanup
- yield from for delegation to sub-generators
"""

from typing import Iterator, List, Any, Union


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

    Hints:
        - Use a list to accumulate items
        - yield the batch when it reaches batch_size
        - Don't forget to yield the final partial batch
    """
    # TODO: Implement batch_generator
    # Hint: Create a batch list, append items, yield when full
    pass


def generator_pipeline(numbers: Iterator[int]) -> Iterator[int]:
    """
    Generator pipeline that chains multiple transformations.

    This demonstrates how generators can be chained together for
    memory-efficient data processing.

    Pipeline stages:
    1. Square each number
    2. Filter for even numbers
    3. Double the result

    Args:
        numbers: Input numbers

    Yields:
        Transformed numbers

    Example:
        >>> list(generator_pipeline(range(5)))
        [0, 8, 32]  # 0^2=0(even)->0, 2^2=4(even)->8, 4^2=16(even)->32

    Hints:
        - Define inner generator functions for each stage
        - Each stage takes an iterator and yields transformed values
        - Chain them by passing one generator to another
        - Return the final generator (don't consume it)
    """
    # TODO: Implement three generator stages
    # Stage 1: square_gen - yields n**2 for each n
    # Stage 2: filter_even - yields only even numbers
    # Stage 3: double_gen - yields n*2 for each n
    # Then chain them together and return the final generator
    pass


def running_average() -> Iterator[float]:
    """
    Stateful generator that computes running average using send().

    This demonstrates bidirectional communication - the generator
    receives values via send() and yields the running average.

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

    Hints:
        - Maintain total and count as local variables
        - Use: value = yield average (receives via send, yields average)
        - First yield returns None (generator must be primed with next())
        - Use infinite loop: while True
        - Update total and count after receiving each value
    """
    # TODO: Implement stateful generator with send()
    # Initialize: total = 0, count = 0, average = None
    # Loop: receive value with yield, update total/count, compute average
    pass


def file_reader_generator(filepath: str) -> Iterator[str]:
    """
    Generator that reads a file line by line with proper cleanup.

    This demonstrates the try/finally pattern for ensuring cleanup
    happens even if the generator is closed early or an exception occurs.

    Args:
        filepath: Path to file to read

    Yields:
        Lines from the file

    Example:
        >>> gen = file_reader_generator("data.txt")
        >>> first_line = next(gen)
        >>> gen.close()  # Cleanup happens in finally block

    Hints:
        - Open file before try block
        - yield lines in try block
        - Close file in finally block
        - finally always executes, even if generator.close() is called
    """
    # TODO: Implement generator with cleanup
    # Open file, use try/finally, yield lines, close in finally
    pass


def flatten_nested(nested: List[Any]) -> Iterator[Any]:
    """
    Recursively flatten a nested list structure using yield from.

    This demonstrates yield from for delegation - it passes all values
    from a sub-generator directly to the caller.

    Args:
        nested: Nested list structure (can contain lists or non-list items)

    Yields:
        Flattened elements in order

    Example:
        >>> list(flatten_nested([1, [2, 3], [4, [5, 6]]]))
        [1, 2, 3, 4, 5, 6]

    Hints:
        - Iterate through items in the list
        - If item is a list, use: yield from flatten_nested(item)
        - If item is not a list, use: yield item
        - yield from completely delegates to the sub-generator
        - This is recursive - base case is non-list items
    """
    # TODO: Implement recursive flattening with yield from
    # Check if item is list: if so, yield from recursive call
    # Otherwise: yield the item directly
    pass
