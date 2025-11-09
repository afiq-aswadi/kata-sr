"""Validated __init_subclass__ - reference solution."""


class ValidatedBase:
    """Base class that validates subclass attributes using __init_subclass__."""

    def __init_subclass__(cls, required_attrs=None, **kwargs):
        """Validate that subclass has required attributes."""
        super().__init_subclass__(**kwargs)
        if required_attrs is not None:
            for attr in required_attrs:
                if attr not in cls.__dict__:
                    raise TypeError(
                        f"Class {cls.__name__} missing required attribute: {attr}"
                    )
