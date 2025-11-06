"""Operator overloading kata."""

from typing import Any


class Vector:
    """2D vector with arithmetic operations."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other):
        """Add two vectors."""
        # TODO: return new Vector with component-wise addition
        # BLANK_START
        pass
        # BLANK_END

    def __sub__(self, other):
        """Subtract two vectors."""
        # TODO: return new Vector with component-wise subtraction
        # BLANK_START
        pass
        # BLANK_END

    def __mul__(self, scalar):
        """Multiply vector by scalar."""
        # TODO: return new Vector with scalar multiplication
        # BLANK_START
        pass
        # BLANK_END

    def __eq__(self, other):
        """Check equality."""
        # TODO: compare x and y components
        # BLANK_START
        pass
        # BLANK_END

    def __repr__(self):
        """String representation."""
        return f"Vector({self.x}, {self.y})"


class Money:
    """Money with currency and arithmetic operations."""

    def __init__(self, amount: float, currency: str = "USD"):
        self.amount = amount
        self.currency = currency

    def __add__(self, other):
        """Add money (must have same currency)."""
        # TODO: check currency, add amounts
        # BLANK_START
        pass
        # BLANK_END

    def __lt__(self, other):
        """Less than comparison."""
        # TODO: check currency, compare amounts
        # BLANK_START
        pass
        # BLANK_END

    def __eq__(self, other):
        """Equality comparison."""
        # TODO: compare currency and amount
        # BLANK_START
        pass
        # BLANK_END

    def __repr__(self):
        return f"{self.amount:.2f} {self.currency}"


class Matrix:
    """Simple 2D matrix with indexing."""

    def __init__(self, data: list[list[float]]):
        self.data = data

    def __getitem__(self, key):
        """Get item using matrix[i, j] syntax."""
        # TODO: handle tuple key (row, col)
        # BLANK_START
        pass
        # BLANK_END

    def __setitem__(self, key, value):
        """Set item using matrix[i, j] = value syntax."""
        # TODO: handle tuple key (row, col)
        # BLANK_START
        pass
        # BLANK_END

    def __len__(self):
        """Return number of rows."""
        # TODO: return len(data)
        # BLANK_START
        pass
        # BLANK_END
