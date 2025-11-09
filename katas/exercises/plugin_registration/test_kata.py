"""Tests for plugin_registration kata."""

import pytest


def test_basic_registration():
    """Test basic plugin registration."""
    from template import PluginBase

    # Clear plugins
    PluginBase._plugins = {}

    class JSONPlugin(PluginBase, plugin_name='json'):
        pass

    assert 'json' in PluginBase._plugins
    assert PluginBase._plugins['json'] is JSONPlugin


def test_multiple_registrations():
    """Test multiple plugins can be registered."""
    from template import PluginBase

    PluginBase._plugins = {}

    class JSONPlugin(PluginBase, plugin_name='json'):
        pass

    class XMLPlugin(PluginBase, plugin_name='xml'):
        pass

    class YAMLPlugin(PluginBase, plugin_name='yaml'):
        pass

    assert len(PluginBase._plugins) == 3
    assert PluginBase._plugins['json'] is JSONPlugin
    assert PluginBase._plugins['xml'] is XMLPlugin
    assert PluginBase._plugins['yaml'] is YAMLPlugin


def test_get_plugin():
    """Test get_plugin retrieves correct plugin."""
    from template import PluginBase

    PluginBase._plugins = {}

    class CSVPlugin(PluginBase, plugin_name='csv'):
        def parse(self):
            return "CSV"

    plugin_class = PluginBase.get_plugin('csv')
    assert plugin_class is CSVPlugin

    plugin = plugin_class()
    assert plugin.parse() == "CSV"


def test_get_plugin_not_found():
    """Test get_plugin returns None for unknown plugin."""
    from template import PluginBase

    PluginBase._plugins = {}

    result = PluginBase.get_plugin('unknown')
    assert result is None


def test_list_plugins():
    """Test list_plugins returns all plugin names."""
    from template import PluginBase

    PluginBase._plugins = {}

    class Plugin1(PluginBase, plugin_name='p1'):
        pass

    class Plugin2(PluginBase, plugin_name='p2'):
        pass

    plugins = PluginBase.list_plugins()
    assert 'p1' in plugins
    assert 'p2' in plugins
    assert len(plugins) == 2


def test_without_name():
    """Test subclass without plugin_name is not registered."""
    from template import PluginBase

    PluginBase._plugins = {}

    class UnnamedPlugin(PluginBase):
        pass

    # Should not be in plugins
    assert len(PluginBase._plugins) == 0


def test_with_inheritance():
    """Test plugins work with inheritance."""
    from template import PluginBase

    PluginBase._plugins = {}

    class BaseParser(PluginBase):
        def base_method(self):
            return "base"

    class JSONParser(BaseParser, plugin_name='json'):
        def parse(self):
            return "JSON"

    assert PluginBase.get_plugin('json') is JSONParser

    parser = JSONParser()
    assert parser.base_method() == "base"
    assert parser.parse() == "JSON"
