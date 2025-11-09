"""Create a simple class dynamically using type() - reference solution."""


def create_simple_class(name: str, attributes: dict) -> type:
    """Create a simple class with given name and attributes."""
    return type(name, (), attributes)
