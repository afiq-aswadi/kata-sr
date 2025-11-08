"""Iterator protocol kata."""

from collections.abc import Iterator
from typing import Any


class Countdown:
    """Iterator that counts down from start to 0."""

    def __init__(self, start: int):
        """Initialize countdown.

        Args:
            start: starting number
        """
        # TODO: save start value and current position
        # BLANK_START
        pass
        # BLANK_END

    def __iter__(self):
        """Return iterator object (self)."""
        # TODO: return self
        # BLANK_START
        pass
        # BLANK_END

    def __next__(self):
        """Return next value or raise StopIteration."""
        # TODO: return current and decrement, raise StopIteration when done
        # BLANK_START
        pass
        # BLANK_END


class InfiniteSequence:
    """Iterator that generates infinite sequence starting from start."""

    def __init__(self, start: int = 0, step: int = 1):
        """Initialize infinite sequence.

        Args:
            start: starting value
            step: increment step
        """
        # TODO: save start and step, initialize current
        # BLANK_START
        pass
        # BLANK_END

    def __iter__(self):
        return self

    def __next__(self):
        """Return next value in sequence."""
        # TODO: return current, then increment by step
        # BLANK_START
        pass
        # BLANK_END


class BatchIterator:
    """Iterator that yields items in batches."""

    def __init__(self, items: list[Any], batch_size: int):
        """Initialize batch iterator.

        Args:
            items: list of items to iterate
            batch_size: number of items per batch
        """
        # TODO: save items, batch_size, current index
        # BLANK_START
        pass
        # BLANK_END

    def __iter__(self):
        return self

    def __next__(self):
        """Return next batch or raise StopIteration."""
        # TODO: return batch of items, raise StopIteration when exhausted
        # BLANK_START
        pass
        # BLANK_END


def fibonacci_generator() -> Iterator[int]:
    """Generator function for Fibonacci sequence.

    Yields:
        next Fibonacci number
    """
    # TODO: implement using generator (yield)
    # BLANK_START
    pass
    # BLANK_END


def take(iterator: Iterator[Any], n: int) -> list[Any]:
    """Take first n items from iterator.

    Args:
        iterator: iterator to take from
        n: number of items to take

    Returns:
        list of first n items
    """
    # TODO: use next() to get n items
    # BLANK_START
    pass
    # BLANK_END
