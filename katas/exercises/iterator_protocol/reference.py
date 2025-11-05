"""Iterator protocol kata - reference solution."""

from typing import Any, Iterator


class Countdown:
    """Iterator that counts down from start to 0."""

    def __init__(self, start: int):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < 0:
            raise StopIteration
        value = self.current
        self.current -= 1
        return value


class InfiniteSequence:
    """Iterator that generates infinite sequence starting from start."""

    def __init__(self, start: int = 0, step: int = 1):
        self.current = start
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        value = self.current
        self.current += self.step
        return value


class BatchIterator:
    """Iterator that yields items in batches."""

    def __init__(self, items: list[Any], batch_size: int):
        self.items = items
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.items):
            raise StopIteration

        batch = self.items[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return batch


def fibonacci_generator() -> Iterator[int]:
    """Generator function for Fibonacci sequence."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def take(iterator: Iterator[Any], n: int) -> list[Any]:
    """Take first n items from iterator."""
    result = []
    for _ in range(n):
        try:
            result.append(next(iterator))
        except StopIteration:
            break
    return result
