"""Tests for registry metaclass kata."""

import pytest


def test_registry_basic():
    """Test that classes are registered."""
    from template import RegistryMeta, _class_registry, clear_registry

    clear_registry()

    class ModelA(metaclass=RegistryMeta):
        pass

    assert 'ModelA' in _class_registry
    assert _class_registry['ModelA'] is ModelA


def test_registry_multiple_classes():
    """Test multiple classes are all registered."""
    from template import RegistryMeta, _class_registry, clear_registry

    clear_registry()

    class Model1(metaclass=RegistryMeta):
        pass

    class Model2(metaclass=RegistryMeta):
        pass

    class Model3(metaclass=RegistryMeta):
        pass

    assert len(_class_registry) == 3
    assert 'Model1' in _class_registry
    assert 'Model2' in _class_registry
    assert 'Model3' in _class_registry


def test_get_registered_classes():
    """Test get_registered_classes returns all classes."""
    from template import RegistryMeta, get_registered_classes, clear_registry

    clear_registry()

    class PluginA(metaclass=RegistryMeta):
        pass

    class PluginB(metaclass=RegistryMeta):
        pass

    registry = get_registered_classes()

    assert 'PluginA' in registry
    assert 'PluginB' in registry
    assert registry['PluginA'] is PluginA
    assert registry['PluginB'] is PluginB


def test_get_class_by_name():
    """Test retrieving a class by name."""
    from template import RegistryMeta, get_class_by_name, clear_registry

    clear_registry()

    class MyModel(metaclass=RegistryMeta):
        value = 42

    retrieved = get_class_by_name('MyModel')

    assert retrieved is MyModel
    assert retrieved.value == 42


def test_get_class_by_name_not_found():
    """Test retrieving non-existent class returns None."""
    from template import get_class_by_name, clear_registry

    clear_registry()

    result = get_class_by_name('NonExistent')
    assert result is None


def test_clear_registry():
    """Test clearing the registry."""
    from template import RegistryMeta, _class_registry, clear_registry

    class Temp(metaclass=RegistryMeta):
        pass

    assert len(_class_registry) > 0

    clear_registry()

    assert len(_class_registry) == 0


def test_registry_with_inheritance():
    """Test registry works with class inheritance."""
    from template import RegistryMeta, _class_registry, clear_registry

    clear_registry()

    class Base(metaclass=RegistryMeta):
        pass

    class Child(Base):
        pass

    # Both should be registered
    assert 'Base' in _class_registry
    assert 'Child' in _class_registry
    assert issubclass(Child, Base)


def test_registry_preserves_attributes():
    """Test that registered classes preserve their attributes."""
    from template import RegistryMeta, get_class_by_name, clear_registry

    clear_registry()

    class User(metaclass=RegistryMeta):
        role = "admin"

        def greet(self):
            return "Hello"

    retrieved = get_class_by_name('User')

    assert retrieved.role == "admin"

    user = retrieved()
    assert user.greet() == "Hello"


def test_registry_with_methods():
    """Test registry works with classes that have methods."""
    from template import RegistryMeta, _class_registry, clear_registry

    clear_registry()

    class Calculator(metaclass=RegistryMeta):
        def add(self, a, b):
            return a + b

    assert 'Calculator' in _class_registry

    calc = _class_registry['Calculator']()
    assert calc.add(2, 3) == 5


def test_registry_instantiation():
    """Test that registered classes can be instantiated."""
    from template import RegistryMeta, get_class_by_name, clear_registry

    clear_registry()

    class Product(metaclass=RegistryMeta):
        def __init__(self, name, price):
            self.name = name
            self.price = price

    ProductClass = get_class_by_name('Product')
    product = ProductClass("Laptop", 999)

    assert product.name == "Laptop"
    assert product.price == 999


def test_registry_class_metadata():
    """Test that class metadata is preserved."""
    from template import RegistryMeta, get_class_by_name, clear_registry

    clear_registry()

    class MyService(metaclass=RegistryMeta):
        """Service documentation."""
        pass

    retrieved = get_class_by_name('MyService')

    assert retrieved.__name__ == 'MyService'
    assert retrieved.__doc__ == "Service documentation."


def test_registry_factory_pattern():
    """Test using registry as a factory pattern."""
    from template import RegistryMeta, get_class_by_name, clear_registry

    clear_registry()

    class JSONParser(metaclass=RegistryMeta):
        def parse(self, data):
            return "JSON"

    class XMLParser(metaclass=RegistryMeta):
        def parse(self, data):
            return "XML"

    # Factory function
    def create_parser(parser_type):
        parser_class = get_class_by_name(f'{parser_type}Parser')
        return parser_class() if parser_class else None

    json_parser = create_parser('JSON')
    xml_parser = create_parser('XML')

    assert json_parser.parse("") == "JSON"
    assert xml_parser.parse("") == "XML"
