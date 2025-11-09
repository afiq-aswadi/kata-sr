"""Create a simple class dynamically using type()."""


def create_simple_class(name: str, attributes: dict) -> type:
    """Create a simple class with given name and attributes.

    Args:
        name: The name of the class to create
        attributes: Dictionary of class attributes and methods

    Returns:
        A new class type

    Example:
        MyClass = create_simple_class('MyClass', {'value': 42})
        obj = MyClass()
        assert obj.value == 42
    """
    # BLANK_START
    raise NotImplementedError("Use type(name, bases, dict) with empty bases tuple")
    # BLANK_END
