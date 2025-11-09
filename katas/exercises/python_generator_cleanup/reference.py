"""
Reference implementation for Python Generator Cleanup kata
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
        >>> gen.close()  # Cleanup happens here
    """
    f = open(filepath, 'r')
    try:
        for line in f:
            yield line
    finally:
        # Cleanup code - always executes
        f.close()
