"""Context manager kata - reference solution."""

import time
from contextlib import contextmanager
from typing import Any


class Timer:
    """Context manager to measure execution time."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        """Start timer."""
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and calculate elapsed time."""
        self.elapsed = time.time() - self.start
        return False


class TemporaryValue:
    """Context manager to temporarily change an object's attribute."""

    def __init__(self, obj: Any, attr: str, value: Any):
        self.obj = obj
        self.attr = attr
        self.value = value
        self.original = getattr(obj, attr)

    def __enter__(self):
        """Set temporary value."""
        setattr(self.obj, self.attr, self.value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original value."""
        setattr(self.obj, self.attr, self.original)
        return False


class SuppressException:
    """Context manager to suppress specific exceptions."""

    def __init__(self, *exceptions):
        self.exceptions = exceptions

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Suppress exceptions if they match."""
        if exc_type is not None and issubclass(exc_type, self.exceptions):
            return True
        return False


@contextmanager
def file_writer(filename: str):
    """Context manager using decorator for file writing."""
    f = open(filename, "w")
    try:
        yield f
    finally:
        f.close()
