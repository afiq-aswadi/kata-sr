"""Metaclass kata - reference solution."""


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


class ValidatedMeta(type):
    """Metaclass that validates required class attributes."""

    def __new__(mcs, name, bases, namespace):
        """Create class and validate required attributes exist."""
        cls = super().__new__(mcs, name, bases, namespace)

        # Skip validation for base class (if it doesn't have required_attributes)
        if 'required_attributes' in namespace:
            required = namespace['required_attributes']
            for attr in required:
                if attr not in namespace:
                    raise TypeError(
                        f"Class {name} missing required attribute: {attr}"
                    )

        return cls


# Global registry for tracking subclasses
_class_registry = {}


class RegistryMeta(type):
    """Metaclass that auto-registers all subclasses."""

    def __new__(mcs, name, bases, namespace):
        """Create class and register it in global registry."""
        cls = super().__new__(mcs, name, bases, namespace)
        _class_registry[name] = cls
        return cls


class AutoPropertyMeta(type):
    """Metaclass that adds automatic property getters/setters."""

    def __new__(mcs, name, bases, namespace):
        """Create class and add properties for attributes in _auto_properties."""
        cls = super().__new__(mcs, name, bases, namespace)

        if '_auto_properties' in namespace:
            for attr_name in namespace['_auto_properties']:
                # Create property with getter and setter
                def make_property(attr):
                    storage_name = f"_{attr}"

                    def getter(self):
                        return getattr(self, storage_name, None)

                    def setter(self, value):
                        setattr(self, storage_name, value)

                    return property(getter, setter)

                setattr(cls, attr_name, make_property(attr_name))

        return cls


class BasePlugin:
    """Base class demonstrating __init_subclass__ as metaclass alternative."""

    _plugins = {}

    def __init_subclass__(cls, plugin_name=None, **kwargs):
        """Register subclass when it's defined."""
        super().__init_subclass__(**kwargs)
        if plugin_name is not None:
            cls._plugins[plugin_name] = cls


def create_class_dynamically(name: str, base_class: type, attributes: dict) -> type:
    """Create a class dynamically using type()."""
    return type(name, (base_class,), attributes)
