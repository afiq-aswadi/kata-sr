"""Auto property metaclass kata - reference solution."""


class AutoPropertyMeta(type):
    """Metaclass that adds automatic property getters/setters."""

    def __new__(mcs, name, bases, namespace):
        """Create class and add properties for attributes in _auto_properties."""
        cls = super().__new__(mcs, name, bases, namespace)

        if '_auto_properties' in namespace:
            for attr_name in namespace['_auto_properties']:
                # Use helper function to avoid closure issues
                prop = make_auto_property(attr_name)
                setattr(cls, attr_name, prop)

        return cls


def make_auto_property(attr_name: str):
    """Helper function to create a property for an attribute."""
    storage_name = f"_{attr_name}"

    def getter(self):
        return getattr(self, storage_name, None)

    def setter(self, value):
        setattr(self, storage_name, value)

    return property(getter, setter)
