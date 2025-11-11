"""Tests for auto property metaclass kata."""

import pytest


try:
    from user_kata import AutoPropertyMeta
    from user_kata import make_auto_property
except ImportError:
    from .reference import AutoPropertyMeta
    from .reference import make_auto_property


def test_auto_property_basic():
    """Test AutoPropertyMeta creates basic properties."""

    class Person(metaclass=AutoPropertyMeta):
        _auto_properties = ['name']

        def __init__(self, name):
            self.name = name

    p = Person("Alice")
    assert p.name == "Alice"


def test_auto_property_get_set():
    """Test property getter and setter work."""

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


def test_auto_property_storage():
    """Test AutoPropertyMeta stores values with underscore prefix."""

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


def test_auto_property_none_default():
    """Test properties return None when not set."""

    class Model(metaclass=AutoPropertyMeta):
        _auto_properties = ['value']

    m = Model()
    assert m.value is None


def test_auto_property_multiple():
    """Test multiple properties work independently."""

    class Point(metaclass=AutoPropertyMeta):
        _auto_properties = ['x', 'y', 'z']

        def __init__(self):
            self.x = 1
            self.y = 2
            self.z = 3

    p = Point()
    assert p.x == 1
    assert p.y == 2
    assert p.z == 3

    p.x = 10
    assert p.x == 10
    assert p.y == 2  # Other properties unchanged


def test_auto_property_no_properties():
    """Test AutoPropertyMeta works without _auto_properties."""

    class Basic(metaclass=AutoPropertyMeta):
        value = 42

    b = Basic()
    assert b.value == 42


def test_auto_property_empty_list():
    """Test AutoPropertyMeta works with empty _auto_properties."""

    class Empty(metaclass=AutoPropertyMeta):
        _auto_properties = []
        value = 100

    e = Empty()
    assert e.value == 100


def test_auto_property_multiple_instances():
    """Test properties maintain separate state per instance."""

    class Counter(metaclass=AutoPropertyMeta):
        _auto_properties = ['count']

        def __init__(self):
            self.count = 0

    c1 = Counter()
    c2 = Counter()

    c1.count = 5
    c2.count = 10

    assert c1.count == 5
    assert c2.count == 10


def test_auto_property_with_methods():
    """Test properties work alongside regular methods."""

    class Rectangle(metaclass=AutoPropertyMeta):
        _auto_properties = ['width', 'height']

        def __init__(self, width, height):
            self.width = width
            self.height = height

        def area(self):
            return self.width * self.height

    r = Rectangle(5, 3)
    assert r.area() == 15

    r.width = 10
    assert r.area() == 30


def test_auto_property_string_values():
    """Test properties work with string values."""

    class User(metaclass=AutoPropertyMeta):
        _auto_properties = ['username', 'email']

    u = User()
    u.username = "alice"
    u.email = "alice@example.com"

    assert u.username == "alice"
    assert u.email == "alice@example.com"


def test_auto_property_list_values():
    """Test properties work with list values."""

    class Container(metaclass=AutoPropertyMeta):
        _auto_properties = ['items']

    c = Container()
    c.items = [1, 2, 3]

    assert c.items == [1, 2, 3]

    c.items.append(4)
    assert c.items == [1, 2, 3, 4]


def test_auto_property_update_value():
    """Test properties can be updated multiple times."""

    class Settings(metaclass=AutoPropertyMeta):
        _auto_properties = ['theme']

    s = Settings()
    s.theme = "light"
    assert s.theme == "light"

    s.theme = "dark"
    assert s.theme == "dark"

    s.theme = "auto"
    assert s.theme == "auto"


def test_auto_property_is_property():
    """Test that created attributes are actual property objects."""

    class Model(metaclass=AutoPropertyMeta):
        _auto_properties = ['value']

    # Check that 'value' is a property on the class
    assert isinstance(Model.__dict__['value'], property)


def test_make_auto_property_helper():
    """Test the helper function creates properties correctly."""

    prop = make_auto_property('test_attr')

    assert isinstance(prop, property)

    # Test the property works
    class Dummy:
        pass

    d = Dummy()
    # Use property's fget and fset
    prop.fset(d, 42)
    assert prop.fget(d) == 42
    assert d._test_attr == 42


def test_auto_property_bool_values():
    """Test properties work with boolean values."""

    class Flags(metaclass=AutoPropertyMeta):
        _auto_properties = ['enabled', 'debug']

    f = Flags()
    f.enabled = True
    f.debug = False

    assert f.enabled is True
    assert f.debug is False
