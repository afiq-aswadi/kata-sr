"""
Reference implementation for Python Generator Send kata
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
    total = 0
    count = 0
    average = None

    while True:
        # Receive value via send()
        value = yield average
        total += value
        count += 1
        average = total / count
