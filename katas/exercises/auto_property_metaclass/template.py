"""Auto property metaclass kata - automatically create properties."""


class AutoPropertyMeta(type):
    """Metaclass that adds automatic property getters/setters.

    Classes should define '_auto_properties' as a list of attribute names
    that should be converted to properties.

    Example:
        class Person(metaclass=AutoPropertyMeta):
            _auto_properties = ['name', 'age']

            def __init__(self, name, age):
                self.name = name  # Uses property setter
                self.age = age

        p = Person("Alice", 30)
        print(p.name)  # Uses property getter
        p.name = "Bob"  # Uses property setter
    """

    def __new__(mcs, name, bases, namespace):
        """Create class and add properties for attributes in _auto_properties.

        For each attribute name in _auto_properties:
        1. Create a property with getter and setter
        2. Getter should return from instance._<attribute_name>
        3. Setter should store to instance._<attribute_name>
        4. Use setattr to add the property to the class

        Args:
            mcs: The metaclass itself
            name: The name of the class being created
            bases: Tuple of base classes
            namespace: Dictionary of class attributes and methods

        Returns:
            The newly created class with properties added
        """
        # TODO: create the class using super().__new__(mcs, name, bases, namespace)
        # TODO: check if '_auto_properties' exists in namespace
        # If it does:
        #   - iterate through each attribute name in _auto_properties
        #   - for each attribute, create a property:
        #     - define a getter function that returns getattr(self, f'_{attr_name}', None)
        #     - define a setter function that calls setattr(self, f'_{attr_name}', value)
        #     - create property using property(getter, setter)
        #     - use setattr(cls, attr_name, property_object) to add to class
        #   - IMPORTANT: use a helper function or closure to avoid variable capture issues
        # TODO: return the created class
        # BLANK_START
        pass
        # BLANK_END


def make_auto_property(attr_name: str):
    """Helper function to create a property for an attribute.

    This helper avoids closure variable capture issues when creating
    multiple properties in a loop.

    Args:
        attr_name: The name of the attribute

    Returns:
        A property object with getter and setter
    """
    # TODO: create storage_name as f'_{attr_name}'
    # TODO: define getter function that returns getattr(self, storage_name, None)
    # TODO: define setter function that calls setattr(self, storage_name, value)
    # TODO: return property(getter, setter)
    # BLANK_START
    pass
    # BLANK_END
