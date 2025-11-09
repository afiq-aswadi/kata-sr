"""Plugin registration using __init_subclass__."""


class PluginBase:
    """Base class for plugins using __init_subclass__ for registration.

    Example:
        class JSONPlugin(PluginBase, plugin_name='json'):
            def parse(self, data):
                return "JSON parsed"

        class XMLPlugin(PluginBase, plugin_name='xml'):
            def parse(self, data):
                return "XML parsed"

        print(PluginBase.get_plugin('json'))  # Returns JSONPlugin class
    """

    _plugins = {}

    def __init_subclass__(cls, plugin_name=None, **kwargs):
        """Called when a subclass is created.

        This hook is automatically called when someone creates a class
        that inherits from PluginBase.

        Args:
            cls: The subclass being created
            plugin_name: Optional name to register the plugin under
            **kwargs: Other keyword arguments (pass to super)
        """
        # BLANK_START
        raise NotImplementedError(
            "Call super().__init_subclass__(**kwargs), register cls if plugin_name given"
        )
        # BLANK_END

    @classmethod
    def get_plugin(cls, name):
        """Get a registered plugin by name."""
        return cls._plugins.get(name)

    @classmethod
    def list_plugins(cls):
        """List all registered plugin names."""
        return list(cls._plugins.keys())
