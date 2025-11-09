"""Configurable __init_subclass__ - store configuration."""


class ConfigurableBase:
    """Base class that stores configuration from subclass kwargs.

    Example:
        class MyService(ConfigurableBase, timeout=30, retries=3):
            pass

        print(MyService.get_config())  # {'timeout': 30, 'retries': 3}
    """

    def __init_subclass__(cls, **config):
        """Store configuration options in the subclass.

        Args:
            cls: The subclass being created
            **config: Configuration key-value pairs
        """
        # BLANK_START
        raise NotImplementedError(
            "Call super().__init_subclass__(), store config as cls._config"
        )
        # BLANK_END

    @classmethod
    def get_config(cls):
        """Get the configuration for this class."""
        return getattr(cls, '_config', {})
