"""Tests for metaclass kata."""

import pytest


def test_singleton_basic():
    """Test singleton metaclass creates only one instance."""
    from template import SingletonMeta

    class Database(metaclass=SingletonMeta):
        def __init__(self):
            self.connection = "connected"

    db1 = Database()
    db2 = Database()

    assert db1 is db2
    assert db1.connection == "connected"


def test_singleton_multiple_classes():
    """Test singleton works independently for different classes."""
    from template import SingletonMeta

    class ServiceA(metaclass=SingletonMeta):
        pass

    class ServiceB(metaclass=SingletonMeta):
        pass

    a1 = ServiceA()
    a2 = ServiceA()
    b1 = ServiceB()
    b2 = ServiceB()

    assert a1 is a2
    assert b1 is b2
    assert a1 is not b1


def test_singleton_with_args():
    """Test singleton ignores constructor args after first instance."""
    from template import SingletonMeta

    class Config(metaclass=SingletonMeta):
        def __init__(self, value=None):
            if not hasattr(self, 'initialized'):
                self.value = value
                self.initialized = True

    c1 = Config(value=42)
    c2 = Config(value=99)  # This should return c1

    assert c1 is c2
    assert c1.value == 42


def test_validated_meta_success():
    """Test ValidatedMeta allows class with all required attributes."""
    from template import ValidatedMeta

    class User(metaclass=ValidatedMeta):
        required_attributes = ['name', 'email']
        name = "John"
        email = "john@example.com"

    u = User()
    assert u.name == "John"
    assert u.email == "john@example.com"


def test_validated_meta_missing_attribute():
    """Test ValidatedMeta raises error for missing required attribute."""
    from template import ValidatedMeta

    with pytest.raises(TypeError, match="missing required attribute"):
        class InvalidUser(metaclass=ValidatedMeta):
            required_attributes = ['name', 'email']
            name = "John"
            # email is missing


def test_validated_meta_no_requirements():
    """Test ValidatedMeta works without required_attributes."""
    from template import ValidatedMeta

    class Product(metaclass=ValidatedMeta):
        price = 100

    p = Product()
    assert p.price == 100


def test_registry_meta_registers_classes():
    """Test RegistryMeta adds classes to global registry."""
    from template import RegistryMeta, _class_registry

    # Clear registry
    _class_registry.clear()

    class ModelA(metaclass=RegistryMeta):
        pass

    class ModelB(metaclass=RegistryMeta):
        pass

    assert 'ModelA' in _class_registry
    assert 'ModelB' in _class_registry
    assert _class_registry['ModelA'] is ModelA
    assert _class_registry['ModelB'] is ModelB


def test_registry_meta_unique_names():
    """Test registry contains all registered class names."""
    from template import RegistryMeta, _class_registry

    _class_registry.clear()

    class Plugin1(metaclass=RegistryMeta):
        pass

    class Plugin2(metaclass=RegistryMeta):
        pass

    class Plugin3(metaclass=RegistryMeta):
        pass

    assert len(_class_registry) >= 3
    assert all(name in _class_registry for name in ['Plugin1', 'Plugin2', 'Plugin3'])


def test_auto_property_meta_basic():
    """Test AutoPropertyMeta creates properties."""
    from template import AutoPropertyMeta

    class Person(metaclass=AutoPropertyMeta):
        _auto_properties = ['name', 'age']

        def __init__(self, name, age):
            self.name = name
            self.age = age

    p = Person("Alice", 30)
    assert p.name == "Alice"
    assert p.age == 30

    p.name = "Bob"
    p.age = 25
    assert p.name == "Bob"
    assert p.age == 25


def test_auto_property_meta_storage():
    """Test AutoPropertyMeta stores values with underscore prefix."""
    from template import AutoPropertyMeta

    class Config(metaclass=AutoPropertyMeta):
        _auto_properties = ['host', 'port']

    c = Config()
    c.host = "localhost"
    c.port = 8080

    # Properties should store with underscore prefix
    assert hasattr(c, '_host')
    assert hasattr(c, '_port')
    assert c._host == "localhost"
    assert c._port == 8080


def test_auto_property_meta_no_properties():
    """Test AutoPropertyMeta works without _auto_properties."""
    from template import AutoPropertyMeta

    class Basic(metaclass=AutoPropertyMeta):
        value = 42

    b = Basic()
    assert b.value == 42


def test_init_subclass_registration():
    """Test __init_subclass__ registers plugins."""
    from template import BasePlugin

    # Clear plugins
    BasePlugin._plugins = {}

    class JsonPlugin(BasePlugin, plugin_name='json'):
        pass

    class XmlPlugin(BasePlugin, plugin_name='xml'):
        pass

    assert 'json' in BasePlugin._plugins
    assert 'xml' in BasePlugin._plugins
    assert BasePlugin._plugins['json'] is JsonPlugin
    assert BasePlugin._plugins['xml'] is XmlPlugin


def test_init_subclass_optional_name():
    """Test __init_subclass__ works without plugin_name."""
    from template import BasePlugin

    BasePlugin._plugins = {}

    class UnnamedPlugin(BasePlugin):
        pass

    # Should not raise error, just not register
    assert 'UnnamedPlugin' not in BasePlugin._plugins


def test_create_class_dynamically_basic():
    """Test dynamic class creation with type()."""
    from template import create_class_dynamically

    class Animal:
        pass

    Dog = create_class_dynamically(
        'Dog',
        Animal,
        {'sound': 'woof', 'legs': 4}
    )

    assert Dog.__name__ == 'Dog'
    assert issubclass(Dog, Animal)
    assert Dog.sound == 'woof'
    assert Dog.legs == 4


def test_create_class_dynamically_with_method():
    """Test dynamic class creation with methods."""
    from template import create_class_dynamically

    class Base:
        pass

    def greet(self):
        return "Hello!"

    MyClass = create_class_dynamically(
        'MyClass',
        Base,
        {'greet': greet, 'value': 42}
    )

    obj = MyClass()
    assert obj.greet() == "Hello!"
    assert obj.value == 42


def test_metaclass_type_hierarchy():
    """Test metaclass is instance of type."""
    from template import SingletonMeta, ValidatedMeta

    assert isinstance(SingletonMeta, type)
    assert isinstance(ValidatedMeta, type)


def test_multiple_inheritance_mro():
    """Test metaclass works with multiple inheritance."""
    from template import RegistryMeta, _class_registry

    _class_registry.clear()

    class Mixin:
        def helper(self):
            return "helping"

    class Service(Mixin, metaclass=RegistryMeta):
        pass

    assert 'Service' in _class_registry
    s = Service()
    assert s.helper() == "helping"


def test_class_metadata_preserved():
    """Test metaclass preserves class name and module."""
    from template import SingletonMeta

    class MyService(metaclass=SingletonMeta):
        """Service documentation."""
        pass

    assert MyService.__name__ == 'MyService'
    assert MyService.__doc__ == "Service documentation."
