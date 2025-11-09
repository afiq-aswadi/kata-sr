"""Validated metaclass - enforce required class attributes."""


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
        # This works!

        class InvalidUser(metaclass=ValidatedMeta):
            required_attributes = ['name', 'email']
            name = "John"
            # Missing email - will raise TypeError!
    """

    def __new__(mcs, name, bases, namespace):
        """Create class and validate required attributes exist.

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
        # BLANK_START
        raise NotImplementedError(
            "Create class with super().__new__, check required_attributes with hasattr(cls, attr)"
        )
        # BLANK_END
