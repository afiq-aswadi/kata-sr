"""Init subclass kata - use __init_subclass__ instead of metaclasses."""


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
        # TODO: call super().__init_subclass__(**kwargs)
        # TODO: if plugin_name is not None:
        #   - register cls in _plugins dict with plugin_name as key
        # BLANK_START
        pass
        # BLANK_END

    @classmethod
    def get_plugin(cls, name):
        """Get a registered plugin by name.

        Args:
            name: The plugin name

        Returns:
            The plugin class, or None if not found
        """
        # TODO: return the plugin from _plugins, or None if not found
        # BLANK_START
        pass
        # BLANK_END

    @classmethod
    def list_plugins(cls):
        """List all registered plugin names.

        Returns:
            List of plugin names
        """
        # TODO: return list of keys from _plugins
        # BLANK_START
        pass
        # BLANK_END


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
        # TODO: call super().__init_subclass__(**kwargs)
        # TODO: if required_attrs is not None:
        #   - iterate through required_attrs
        #   - check if each attr exists in cls.__dict__
        #   - if missing, raise TypeError with message:
        #     f"Class {cls.__name__} missing required attribute: {attr}"
        # BLANK_START
        pass
        # BLANK_END


class ConfigurableBase:
    """Base class that stores configuration from subclass kwargs.

    Example:
        class MyService(ConfigurableBase, timeout=30, retries=3):
            pass

        print(MyService._config)  # {'timeout': 30, 'retries': 3}
    """

    def __init_subclass__(cls, **config):
        """Store configuration options in the subclass.

        Args:
            cls: The subclass being created
            **config: Configuration key-value pairs
        """
        # TODO: call super().__init_subclass__()
        # TODO: store config dict as cls._config
        # BLANK_START
        pass
        # BLANK_END

    @classmethod
    def get_config(cls):
        """Get the configuration for this class.

        Returns:
            Configuration dictionary
        """
        # TODO: return cls._config if it exists, else empty dict
        # BLANK_START
        pass
        # BLANK_END
