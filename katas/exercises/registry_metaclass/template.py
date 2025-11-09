"""Registry metaclass kata - automatically register all subclasses."""

# Global registry to track all classes created with RegistryMeta
_class_registry = {}


class RegistryMeta(type):
    """Metaclass that auto-registers all classes in a global registry.

    Example:
        class ModelA(metaclass=RegistryMeta):
            pass

        class ModelB(metaclass=RegistryMeta):
            pass

        print(_class_registry)
        # {'ModelA': <class 'ModelA'>, 'ModelB': <class 'ModelB'>}
    """

    def __new__(mcs, name, bases, namespace):
        """Create class and register it in the global registry.

        This is called when a new class is being created (at class definition time).
        Use this to create the class and add it to the registry.

        Args:
            mcs: The metaclass itself (RegistryMeta)
            name: The name of the class being created
            bases: Tuple of base classes
            namespace: Dictionary of class attributes and methods

        Returns:
            The newly created class
        """
        # TODO: create the class using super().__new__(mcs, name, bases, namespace)
        # TODO: add the class to _class_registry with name as the key
        # TODO: return the created class
        # BLANK_START
        pass
        # BLANK_END


def get_registered_classes():
    """Return a dictionary of all registered classes.

    Returns:
        Dictionary mapping class names to class objects
    """
    # TODO: return the _class_registry
    # BLANK_START
    pass
    # BLANK_END


def get_class_by_name(name: str):
    """Get a registered class by its name.

    Args:
        name: The name of the class to retrieve

    Returns:
        The class object, or None if not found
    """
    # TODO: return the class from _class_registry, or None if not found
    # BLANK_START
    pass
    # BLANK_END


def clear_registry():
    """Clear all registered classes.

    Useful for testing or resetting the registry.
    """
    # TODO: clear the _class_registry dict
    # BLANK_START
    pass
    # BLANK_END
