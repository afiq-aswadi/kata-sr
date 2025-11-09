"""Validated metaclass kata - reference solution."""


class ValidatedMeta(type):
    """Metaclass that validates required class attributes."""

    def __new__(mcs, name, bases, namespace):
        """Create class and validate required attributes exist."""
        cls = super().__new__(mcs, name, bases, namespace)

        # Skip validation for base class (if it doesn't have required_attributes)
        if 'required_attributes' in namespace:
            required = namespace['required_attributes']
            for attr in required:
                if attr not in namespace:
                    raise TypeError(
                        f"Class {name} missing required attribute: {attr}"
                    )

        return cls


class TypedMeta(type):
    """Metaclass that validates attribute types."""

    def __new__(mcs, name, bases, namespace):
        """Create class and validate attribute types."""
        cls = super().__new__(mcs, name, bases, namespace)

        if 'attribute_types' in namespace:
            attribute_types = namespace['attribute_types']
            for attr, expected_type in attribute_types.items():
                if attr in namespace:
                    value = namespace[attr]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Attribute {attr} must be {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

        return cls
