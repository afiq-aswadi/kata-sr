"""Create a class with __init__ constructor using type()."""


def create_class_with_init(name: str, init_params: list) -> type:
    """Create a class with an __init__ method that accepts parameters.

    Args:
        name: The name of the class to create
        init_params: List of parameter names to store as instance attributes

    Returns:
        A new class with __init__ method

    Example:
        Person = create_class_with_init('Person', ['name', 'age'])
        p = Person('Alice', 30)
        assert p.name == 'Alice'
        assert p.age == 30
    """
    # BLANK_START
    raise NotImplementedError(
        "Define __init__(self, *args, **kwargs) that stores params as attributes"
    )
    # BLANK_END
