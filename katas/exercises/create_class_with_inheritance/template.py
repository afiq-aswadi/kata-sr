"""Create a class with inheritance using type()."""


def create_class_with_inheritance(name: str, base_class: type, attributes: dict) -> type:
    """Create a class that inherits from a base class.

    Args:
        name: The name of the class to create
        base_class: The parent class to inherit from
        attributes: Dictionary of class attributes and methods

    Returns:
        A new class type that inherits from base_class

    Example:
        class Animal:
            kingdom = "Animalia"

        Dog = create_class_with_inheritance('Dog', Animal, {'sound': 'woof'})
        assert Dog.kingdom == "Animalia"  # Inherited
        assert Dog.sound == 'woof'  # Own attribute
    """
    # BLANK_START
    raise NotImplementedError("Use type(name, (base_class,), attributes)")
    # BLANK_END
