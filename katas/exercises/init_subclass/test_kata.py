"""Tests for init subclass kata."""

import pytest


def test_plugin_basic_registration():
    """Test basic plugin registration."""
    from template import PluginBase

    # Clear plugins
    PluginBase._plugins = {}

    class JSONPlugin(PluginBase, plugin_name='json'):
        pass

    assert 'json' in PluginBase._plugins
    assert PluginBase._plugins['json'] is JSONPlugin


def test_plugin_multiple_registrations():
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


def test_plugin_get_plugin():
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


def test_plugin_get_plugin_not_found():
    """Test get_plugin returns None for unknown plugin."""
    from template import PluginBase

    PluginBase._plugins = {}

    result = PluginBase.get_plugin('unknown')
    assert result is None


def test_plugin_list_plugins():
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


def test_plugin_without_name():
    """Test subclass without plugin_name is not registered."""
    from template import PluginBase

    PluginBase._plugins = {}

    class UnnamedPlugin(PluginBase):
        pass

    # Should not be in plugins
    assert len(PluginBase._plugins) == 0


def test_plugin_inheritance():
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


def test_validated_base_success():
    """Test ValidatedBase allows class with required attributes."""
    from template import ValidatedBase

    class User(ValidatedBase, required_attrs=['name', 'email']):
        name = "John"
        email = "john@example.com"

    u = User()
    assert u.name == "John"
    assert u.email == "john@example.com"


def test_validated_base_missing_attribute():
    """Test ValidatedBase raises error for missing attribute."""
    from template import ValidatedBase

    with pytest.raises(TypeError, match="missing required attribute: email"):
        class BadUser(ValidatedBase, required_attrs=['name', 'email']):
            name = "John"
            # email is missing


def test_validated_base_no_validation():
    """Test ValidatedBase works without required_attrs."""
    from template import ValidatedBase

    class Simple(ValidatedBase):
        value = 42

    s = Simple()
    assert s.value == 42


def test_validated_base_multiple_attrs():
    """Test ValidatedBase validates multiple attributes."""
    from template import ValidatedBase

    class Model(ValidatedBase, required_attrs=['a', 'b', 'c']):
        a = 1
        b = 2
        c = 3

    m = Model()
    assert m.a == 1
    assert m.b == 2
    assert m.c == 3


def test_validated_base_inherited_attrs():
    """Test ValidatedBase can see inherited attributes."""
    from template import ValidatedBase

    class Base(ValidatedBase):
        name = "base"

    class Child(Base, required_attrs=['name']):
        pass  # name is inherited

    # This should work because name is inherited
    # Note: This test depends on implementation - checking __dict__ vs hasattr
    c = Child()
    assert hasattr(c, 'name')


def test_configurable_base_basic():
    """Test ConfigurableBase stores configuration."""
    from template import ConfigurableBase

    class MyService(ConfigurableBase, timeout=30, retries=3):
        pass

    config = MyService.get_config()
    assert config['timeout'] == 30
    assert config['retries'] == 3


def test_configurable_base_no_config():
    """Test ConfigurableBase works without configuration."""
    from template import ConfigurableBase

    class Plain(ConfigurableBase):
        pass

    config = Plain.get_config()
    assert config == {}


def test_configurable_base_empty_config():
    """Test ConfigurableBase with empty configuration."""
    from template import ConfigurableBase

    class Empty(ConfigurableBase):
        pass

    assert Empty.get_config() == {}


def test_configurable_base_multiple_classes():
    """Test different classes maintain separate configs."""
    from template import ConfigurableBase

    class ServiceA(ConfigurableBase, port=8000):
        pass

    class ServiceB(ConfigurableBase, port=9000):
        pass

    assert ServiceA.get_config()['port'] == 8000
    assert ServiceB.get_config()['port'] == 9000


def test_configurable_base_various_types():
    """Test ConfigurableBase handles various config types."""
    from template import ConfigurableBase

    class Complex(ConfigurableBase,
                  name="service",
                  enabled=True,
                  count=5,
                  items=['a', 'b']):
        pass

    config = Complex.get_config()
    assert config['name'] == "service"
    assert config['enabled'] is True
    assert config['count'] == 5
    assert config['items'] == ['a', 'b']


def test_combined_plugin_and_config():
    """Test combining PluginBase and ConfigurableBase patterns."""
    from template import PluginBase, ConfigurableBase

    PluginBase._plugins = {}

    class PluginWithConfig(PluginBase, ConfigurableBase):
        pass

    class MyPlugin(PluginWithConfig, plugin_name='mine', timeout=60):
        pass

    # Should be registered as plugin
    assert PluginBase.get_plugin('mine') is MyPlugin

    # Should have config
    assert MyPlugin.get_config()['timeout'] == 60
