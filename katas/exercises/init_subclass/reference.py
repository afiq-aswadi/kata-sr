"""Init subclass kata - reference solution."""


class PluginBase:
    """Base class for plugins using __init_subclass__ for registration."""

    _plugins = {}

    def __init_subclass__(cls, plugin_name=None, **kwargs):
        """Called when a subclass is created."""
        super().__init_subclass__(**kwargs)
        if plugin_name is not None:
            cls._plugins[plugin_name] = cls

    @classmethod
    def get_plugin(cls, name):
        """Get a registered plugin by name."""
        return cls._plugins.get(name)

    @classmethod
    def list_plugins(cls):
        """List all registered plugin names."""
        return list(cls._plugins.keys())


class ValidatedBase:
    """Base class that validates subclass attributes using __init_subclass__."""

    def __init_subclass__(cls, required_attrs=None, **kwargs):
        """Validate that subclass has required attributes."""
        super().__init_subclass__(**kwargs)
        if required_attrs is not None:
            for attr in required_attrs:
                if attr not in cls.__dict__:
                    raise TypeError(
                        f"Class {cls.__name__} missing required attribute: {attr}"
                    )


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
