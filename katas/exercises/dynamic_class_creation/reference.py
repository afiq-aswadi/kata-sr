"""Dynamic class creation kata - reference solution."""


def create_simple_class(name: str, attributes: dict) -> type:
    """Create a simple class with given name and attributes."""
    return type(name, (), attributes)


def create_class_with_base(name: str, base_class: type, attributes: dict) -> type:
    """Create a class that inherits from a base class."""
    return type(name, (base_class,), attributes)


def create_class_with_methods(name: str, method_dict: dict) -> type:
    """Create a class with methods defined in method_dict."""
    return type(name, (), method_dict)


def create_class_with_init(name: str, init_params: list) -> type:
    """Create a class with an __init__ method that accepts parameters."""

    def __init__(self, *args, **kwargs):
        # Handle positional arguments
        for i, param in enumerate(init_params):
            if i < len(args):
                setattr(self, param, args[i])
            elif param in kwargs:
                setattr(self, param, kwargs[param])

    return type(name, (), {'__init__': __init__})
