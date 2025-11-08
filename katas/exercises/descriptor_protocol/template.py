"""Descriptor protocol kata."""



class ValidatedAttribute:
    """Descriptor that validates attribute values."""

    def __init__(self, min_value: int | None = None, max_value: int | None = None):
        """Initialize validated attribute.

        Args:
            min_value: minimum allowed value
            max_value: maximum allowed value
        """
        # TODO: save min/max values and create storage name
        # BLANK_START
        pass
        # BLANK_END

    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to class attribute."""
        # TODO: save attribute name for storage
        # BLANK_START
        pass
        # BLANK_END

    def __get__(self, obj, objtype=None):
        """Get attribute value."""
        # TODO: return value from obj's __dict__, handle obj=None
        # BLANK_START
        pass
        # BLANK_END

    def __set__(self, obj, value):
        """Set attribute value with validation."""
        # TODO: validate value, then store in obj's __dict__
        # BLANK_START
        pass
        # BLANK_END


class CachedProperty:
    """Descriptor that caches computed property value."""

    def __init__(self, func):
        """Initialize cached property.

        Args:
            func: function to compute value
        """
        # TODO: save function
        # BLANK_START
        pass
        # BLANK_END

    def __set_name__(self, owner, name):
        # TODO: save name for cache key
        # BLANK_START
        pass
        # BLANK_END

    def __get__(self, obj, objtype=None):
        """Get value, computing and caching if needed."""
        # TODO: check cache, compute if missing, store result
        # BLANK_START
        pass
        # BLANK_END


class TypedAttribute:
    """Descriptor that enforces type checking."""

    def __init__(self, expected_type: type):
        """Initialize typed attribute.

        Args:
            expected_type: expected type for values
        """
        # TODO: save expected type
        # BLANK_START
        pass
        # BLANK_END

    def __set_name__(self, owner, name):
        # TODO: save name for storage
        # BLANK_START
        pass
        # BLANK_END

    def __get__(self, obj, objtype=None):
        # TODO: get value from obj's __dict__
        # BLANK_START
        pass
        # BLANK_END

    def __set__(self, obj, value):
        """Set value with type checking."""
        # TODO: check type, raise TypeError if wrong
        # BLANK_START
        pass
        # BLANK_END
