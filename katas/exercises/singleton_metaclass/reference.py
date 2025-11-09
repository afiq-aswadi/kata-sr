"""Singleton metaclass kata - reference solution."""


class SingletonMeta(type):
    """Metaclass that ensures only one instance of a class exists."""

    def __init__(cls, name, bases, namespace):
        """Initialize the class and prepare singleton storage."""
        super().__init__(name, bases, namespace)
        cls._instances = {}

    def __call__(cls, *args, **kwargs):
        """Control instance creation to enforce singleton pattern."""
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
