"""Dynamic class creation kata - create classes at runtime using type()."""


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
    # TODO: use type(name, bases, dict) to create a class
    # bases should be an empty tuple () for no inheritance
    # dict should be the attributes parameter
    # BLANK_START
    pass
    # BLANK_END


def create_class_with_base(name: str, base_class: type, attributes: dict) -> type:
    """Create a class that inherits from a base class.

    Args:
        name: The name of the class to create
        base_class: The parent class to inherit from
        attributes: Dictionary of class attributes and methods

    Returns:
        A new class type that inherits from base_class
    """
    # TODO: use type(name, bases, dict) to create a class
    # bases should be a tuple containing base_class
    # BLANK_START
    pass
    # BLANK_END


def create_class_with_methods(name: str, method_dict: dict) -> type:
    """Create a class with methods defined in method_dict.

    Args:
        name: The name of the class to create
        method_dict: Dictionary mapping method names to functions

    Returns:
        A new class type with the specified methods

    Example:
        def greet(self):
            return "Hello!"

        MyClass = create_class_with_methods('MyClass', {'greet': greet})
        obj = MyClass()
        assert obj.greet() == "Hello!"
    """
    # TODO: create class with methods from method_dict
    # BLANK_START
    pass
    # BLANK_END


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
    # TODO: create an __init__ function that stores parameters as attributes
    # Then create class with this __init__ method
    # Hint: use **kwargs and setattr()
    # BLANK_START
    pass
    # BLANK_END
