"""Context manager kata."""

import time
from contextlib import contextmanager
from typing import Any


class Timer:
    """Context manager to measure execution time."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        """Start timer."""
        # TODO: record start time, return self
        # BLANK_START
        pass
        # BLANK_END

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and calculate elapsed time."""
        # TODO: calculate elapsed time
        # BLANK_START
        pass
        # BLANK_END


class TemporaryValue:
    """Context manager to temporarily change an object's attribute."""

    def __init__(self, obj: Any, attr: str, value: Any):
        """Initialize temporary value context manager.

        Args:
            obj: object to modify
            attr: attribute name
            value: temporary value
        """
        # TODO: save obj, attr, value, and original value
        # BLANK_START
        pass
        # BLANK_END

    def __enter__(self):
        """Set temporary value."""
        # TODO: set attribute to new value
        # BLANK_START
        pass
        # BLANK_END

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original value."""
        # TODO: restore original attribute value
        # BLANK_START
        pass
        # BLANK_END


class SuppressException:
    """Context manager to suppress specific exceptions."""

    def __init__(self, *exceptions):
        """Initialize with exception types to suppress.

        Args:
            exceptions: exception types to suppress
        """
        # TODO: save exception types
        # BLANK_START
        pass
        # BLANK_END

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Suppress exceptions if they match."""
        # TODO: return True if exc_type in exceptions (to suppress)
        # BLANK_START
        pass
        # BLANK_END


@contextmanager
def file_writer(filename: str):
    """Context manager using decorator for file writing.

    Args:
        filename: file to write to

    Yields:
        file handle
    """
    # TODO: open file, yield handle, ensure close in finally
    # BLANK_START
    pass
    # BLANK_END
