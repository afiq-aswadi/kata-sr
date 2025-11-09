"""Plugin registration using __init_subclass__ - reference solution."""


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
