"""
Python Generator Cleanup Kata

Learn how to ensure cleanup code runs in generators using try/finally.

Your task:
Implement a generator that reads a file line by line and ensures the file
is properly closed even if the generator is closed early or an exception occurs.

Key concepts:
- try/finally ensures cleanup code always runs
- finally executes when generator.close() is called
- finally executes when generator exits normally
- finally executes when an exception occurs
- Critical for resource management (files, connections, locks)
"""

from typing import Iterator


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
        >>> gen.close()  # Cleanup happens in finally block

    """
    # BLANK_START
    raise NotImplementedError("Open file, use try/finally, yield lines in try, close file in finally")
    # BLANK_END
