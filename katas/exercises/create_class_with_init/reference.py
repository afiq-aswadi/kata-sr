"""Create a class with __init__ constructor using type() - reference solution."""


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
