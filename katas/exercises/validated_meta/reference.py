"""Validated metaclass - reference solution."""


class ValidatedMeta(type):
    """Metaclass that validates required class attributes."""

    def __new__(mcs, name, bases, namespace):
        """Create class and validate required attributes exist."""
        cls = super().__new__(mcs, name, bases, namespace)

        # Skip validation for base class (if it doesn't have required_attributes)
        if 'required_attributes' in namespace:
            required = namespace['required_attributes']
            for attr in required:
                if not hasattr(cls, attr):
                    raise TypeError(
                        f"Class {name} missing required attribute: {attr}"
                    )

        return cls
