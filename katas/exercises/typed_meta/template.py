"""Typed metaclass - validate attribute types."""


class TypedMeta(type):
    """Metaclass that validates attribute types.

    Classes should define 'attribute_types' as a dict mapping
    attribute names to their expected types.

    Example:
        class Person(metaclass=TypedMeta):
            attribute_types = {'name': str, 'age': int}
            name = "Alice"
            age = 30
        # This works!

        class BadPerson(metaclass=TypedMeta):
            attribute_types = {'age': int}
            age = "thirty"  # Wrong type - raises TypeError!
    """

    def __new__(mcs, name, bases, namespace):
        """Create class and validate attribute types.

        Args:
            mcs: The metaclass itself
            name: The name of the class being created
            bases: Tuple of base classes
            namespace: Dictionary of class attributes and methods

        Returns:
            The newly created class

        Raises:
            TypeError: If any attribute has the wrong type
        """
        # BLANK_START
        raise NotImplementedError(
            "Create class, check attribute_types dict, validate with isinstance"
        )
        # BLANK_END
