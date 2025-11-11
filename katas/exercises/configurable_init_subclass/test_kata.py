"""Tests for configurable_init_subclass kata."""

import pytest


try:
    from user_kata import ConfigurableBase
except ImportError:
    from .reference import ConfigurableBase


def test_basic_configuration():
    """Test ConfigurableBase stores configuration."""

    class MyService(ConfigurableBase, timeout=30, retries=3):
        pass

    config = MyService.get_config()
    assert config['timeout'] == 30
    assert config['retries'] == 3


def test_no_configuration():
    """Test ConfigurableBase works without configuration."""

    class Plain(ConfigurableBase):
        pass

    config = Plain.get_config()
    assert config == {}


def test_multiple_classes():
    """Test different classes maintain separate configs."""

    class ServiceA(ConfigurableBase, port=8000):
        pass

    class ServiceB(ConfigurableBase, port=9000):
        pass

    assert ServiceA.get_config()['port'] == 8000
    assert ServiceB.get_config()['port'] == 9000


def test_various_types():
    """Test ConfigurableBase handles various config types."""

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


def test_empty_configuration():
    """Test ConfigurableBase with no config kwargs."""

    class Empty(ConfigurableBase):
        value = 42

    assert Empty.get_config() == {}
    assert Empty.value == 42


def test_string_config():
    """Test configuration with string values."""

    class App(ConfigurableBase, host="localhost", db="mydb"):
        pass

    config = App.get_config()
    assert config['host'] == "localhost"
    assert config['db'] == "mydb"


def test_single_config_item():
    """Test with single configuration item."""

    class Worker(ConfigurableBase, workers=4):
        pass

    assert Worker.get_config()['workers'] == 4
