"""Create a class with inheritance using type() - reference solution."""


def create_class_with_inheritance(name: str, base_class: type, attributes: dict) -> type:
    """Create a class that inherits from a base class."""
    return type(name, (base_class,), attributes)
