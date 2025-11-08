"""Descriptor protocol kata - reference solution."""



class ValidatedAttribute:
    """Descriptor that validates attribute values."""

    def __init__(self, min_value: int | None = None, max_value: int | None = None):
        self.min_value = min_value
        self.max_value = max_value
        self.name = None

    def __set_name__(self, owner, name):
        self.name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.name, None)

    def __set__(self, obj, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"Value must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"Value must be <= {self.max_value}")
        setattr(obj, self.name, value)


class CachedProperty:
    """Descriptor that caches computed property value."""

    def __init__(self, func):
        self.func = func
        self.name = None

    def __set_name__(self, owner, name):
        self.name = f"_cached_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        # Check if cached
        cached = getattr(obj, self.name, None)
        if cached is not None:
            return cached

        # Compute and cache
        value = self.func(obj)
        setattr(obj, self.name, value)
        return value


class TypedAttribute:
    """Descriptor that enforces type checking."""

    def __init__(self, expected_type: type):
        self.expected_type = expected_type
        self.name = None

    def __set_name__(self, owner, name):
        self.name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.name, None)

    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"Expected {self.expected_type.__name__}, got {type(value).__name__}"
            )
        setattr(obj, self.name, value)
