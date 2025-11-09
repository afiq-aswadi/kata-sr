"""
Python Generator Send Kata

Learn bidirectional communication with generators using send().

Your task:
Implement a stateful generator that computes a running average. It receives
values via send() and yields the current running average after each value.

Key concepts:
- send() passes a value into the generator at the yield point
- Generator must be primed with next() before first send()
- Local variables maintain state between yields
- Useful for stream processing and online algorithms
"""

from typing import Iterator


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
    # BLANK_START
    raise NotImplementedError("Initialize state, use while True loop with 'value = yield average', update total/count")
    # BLANK_END
