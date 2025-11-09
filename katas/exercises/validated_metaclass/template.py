"""Validated metaclass kata - enforce required class attributes."""


class ValidatedMeta(type):
    """Metaclass that validates required class attributes.

    Classes using this metaclass should define a 'required_attributes' list.
    The metaclass will check that all required attributes are present when
    the class is created.

    Example:
        class User(metaclass=ValidatedMeta):
            required_attributes = ['name', 'email']
            name = "John"
            email = "john@example.com"

        # This works fine!

        class InvalidUser(metaclass=ValidatedMeta):
            required_attributes = ['name', 'email']
            name = "John"
            # Missing email - will raise TypeError!
    """

    def __new__(mcs, name, bases, namespace):
        """Create class and validate required attributes exist.

        This is called when a new class is being created.
        Check if required_attributes exists, and validate all are present.

        Args:
            mcs: The metaclass itself (ValidatedMeta)
            name: The name of the class being created
            bases: Tuple of base classes
            namespace: Dictionary of class attributes and methods

        Returns:
            The newly created class

        Raises:
            TypeError: If any required attribute is missing
        """
        # TODO: create the class using super().__new__(mcs, name, bases, namespace)
        # TODO: check if 'required_attributes' exists in namespace
        # If it does:
        #   - iterate through each required attribute
        #   - check if it exists in namespace
        #   - if missing, raise TypeError with message:
        #     f"Class {name} missing required attribute: {attr}"
        # TODO: return the created class
        # BLANK_START
        pass
        # BLANK_END


class TypedMeta(type):
    """Metaclass that validates attribute types.

    Classes should define 'attribute_types' as a dict mapping
    attribute names to their expected types.

    Example:
        class Person(metaclass=TypedMeta):
            attribute_types = {'name': str, 'age': int}
            name = "Alice"
            age = 30
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
        # TODO: create the class using super().__new__
        # TODO: check if 'attribute_types' exists in namespace
        # If it does:
        #   - iterate through attribute_types items
        #   - check if attribute exists in namespace
        #   - if it exists, check if it's an instance of expected type
        #   - if wrong type, raise TypeError with message:
        #     f"Attribute {attr} must be {expected_type.__name__}, got {type(value).__name__}"
        # TODO: return the created class
        # BLANK_START
        pass
        # BLANK_END
