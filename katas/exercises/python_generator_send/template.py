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

    Hints:
        - Initialize total = 0, count = 0, average = None
        - Use infinite loop: while True
        - Use: value = yield average
          - This yields the current average
          - Then receives the next value via send()
        - Update total and count after receiving each value
        - Compute new average: total / count
        - First yield returns None (before any values sent)
    """
    # TODO: Implement stateful generator with send()
    #
    # 1. Initialize: total = 0, count = 0, average = None
    # 2. Start infinite loop (while True)
    # 3. Use: value = yield average
    #    - Yields current average to caller
    #    - Receives new value from send()
    # 4. Update: total += value, count += 1
    # 5. Compute: average = total / count
    # 6. Loop continues, yielding new average
    pass
