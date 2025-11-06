"""Operator overloading kata - reference solution."""

from typing import Any


class Vector:
    """2D vector with arithmetic operations."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other):
        """Add two vectors."""
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """Subtract two vectors."""
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        """Multiply vector by scalar."""
        return Vector(self.x * scalar, self.y * scalar)

    def __eq__(self, other):
        """Check equality."""
        return self.x == other.x and self.y == other.y

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
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def __lt__(self, other):
        """Less than comparison."""
        if self.currency != other.currency:
            raise ValueError(f"Cannot compare {self.currency} and {other.currency}")
        return self.amount < other.amount

    def __eq__(self, other):
        """Equality comparison."""
        return self.amount == other.amount and self.currency == other.currency

    def __repr__(self):
        return f"{self.amount:.2f} {self.currency}"


class Matrix:
    """Simple 2D matrix with indexing."""

    def __init__(self, data: list[list[float]]):
        self.data = data

    def __getitem__(self, key):
        """Get item using matrix[i, j] syntax."""
        if isinstance(key, tuple):
            row, col = key
            return self.data[row][col]
        return self.data[key]

    def __setitem__(self, key, value):
        """Set item using matrix[i, j] = value syntax."""
        if isinstance(key, tuple):
            row, col = key
            self.data[row][col] = value
        else:
            self.data[key] = value

    def __len__(self):
        """Return number of rows."""
        return len(self.data)
