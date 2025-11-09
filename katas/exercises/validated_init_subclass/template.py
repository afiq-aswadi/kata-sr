"""Validated __init_subclass__ - validate required attributes."""


class ValidatedBase:
    """Base class that validates subclass attributes using __init_subclass__.

    Example:
        class User(ValidatedBase, required_attrs=['name', 'email']):
            name = "John"
            email = "john@example.com"
        # This works!

        class BadUser(ValidatedBase, required_attrs=['name']):
            pass  # Missing 'name' - raises TypeError!
    """

    def __init_subclass__(cls, required_attrs=None, **kwargs):
        """Validate that subclass has required attributes.

        Args:
            cls: The subclass being created
            required_attrs: List of required attribute names
            **kwargs: Other keyword arguments
        """
        # BLANK_START
        raise NotImplementedError(
            "Call super().__init_subclass__(**kwargs), check required_attrs with hasattr(cls, attr)"
        )
        # BLANK_END
