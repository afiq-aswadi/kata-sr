"""Configurable __init_subclass__ - reference solution."""


class ConfigurableBase:
    """Base class that stores configuration from subclass kwargs."""

    def __init_subclass__(cls, **config):
        """Store configuration options in the subclass."""
        super().__init_subclass__()
        cls._config = config

    @classmethod
    def get_config(cls):
        """Get the configuration for this class."""
        return getattr(cls, '_config', {})
