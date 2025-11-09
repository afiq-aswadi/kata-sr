"""Create a class with methods using type() - reference solution."""


def create_class_with_methods(name: str, method_dict: dict) -> type:
    """Create a class with methods defined in method_dict."""
    return type(name, (), method_dict)
