"""Registry metaclass kata - reference solution."""

# Global registry to track all classes created with RegistryMeta
_class_registry = {}


class RegistryMeta(type):
    """Metaclass that auto-registers all classes in a global registry."""

    def __new__(mcs, name, bases, namespace):
        """Create class and register it in the global registry."""
        cls = super().__new__(mcs, name, bases, namespace)
        _class_registry[name] = cls
        return cls


def get_registered_classes():
    """Return a dictionary of all registered classes."""
    return _class_registry


def get_class_by_name(name: str):
    """Get a registered class by its name."""
    return _class_registry.get(name)


def clear_registry():
    """Clear all registered classes."""
    _class_registry.clear()
