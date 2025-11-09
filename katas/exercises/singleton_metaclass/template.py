"""Singleton metaclass kata - ensure only one instance exists."""


class SingletonMeta(type):
    """Metaclass that ensures only one instance of a class exists.

    Example:
        class Database(metaclass=SingletonMeta):
            def __init__(self):
                self.connection = "connected"

        db1 = Database()
        db2 = Database()
        assert db1 is db2  # Same instance!
    """

    def __init__(cls, name, bases, namespace):
        """Initialize the class and prepare singleton storage.

        This is called when the class is created (at class definition time).
        Use this to set up the storage for singleton instances.
        """
        # TODO: call super().__init__ with the same parameters
        # TODO: initialize cls._instances as an empty dict
        # BLANK_START
        pass
        # BLANK_END

    def __call__(cls, *args, **kwargs):
        """Control instance creation to enforce singleton pattern.

        This is called when someone does MyClass() to create an instance.
        Check if an instance already exists; if not, create it.

        Returns:
            The single instance of the class
        """
        # TODO: check if cls exists in cls._instances
        # If not, create instance using super().__call__(*args, **kwargs)
        # Store it in cls._instances[cls]
        # Return the instance from cls._instances[cls]
        # BLANK_START
        pass
        # BLANK_END
