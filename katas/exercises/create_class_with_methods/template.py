"""Create a class with methods using type()."""


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
    # BLANK_START
    raise NotImplementedError("Methods are just attributes - use type(name, (), method_dict)")
    # BLANK_END
