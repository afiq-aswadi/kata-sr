"""Tests for create_simple_class kata."""

import pytest


def test_basic_class_creation():
    """Test creating a simple class with attributes."""
    from template import create_simple_class

    MyClass = create_simple_class('MyClass', {'value': 42, 'name': 'test'})

    assert MyClass.__name__ == 'MyClass'
    assert MyClass.value == 42
    assert MyClass.name == 'test'


def test_class_instantiation():
    """Test that dynamically created class can be instantiated."""
    from template import create_simple_class

    MyClass = create_simple_class('MyClass', {'default': 100})

    obj = MyClass()
    assert obj.default == 100


def test_empty_attributes():
    """Test creating class with no attributes."""
    from template import create_simple_class

    EmptyClass = create_simple_class('EmptyClass', {})

    assert EmptyClass.__name__ == 'EmptyClass'
    obj = EmptyClass()
    assert isinstance(obj, EmptyClass)


def test_multiple_attributes():
    """Test creating class with many attributes."""
    from template import create_simple_class

    attrs = {
        'x': 1,
        'y': 2,
        'z': 3,
        'label': 'point'
    }
    Point = create_simple_class('Point', attrs)

    assert Point.x == 1
    assert Point.y == 2
    assert Point.z == 3
    assert Point.label == 'point'


def test_class_is_type():
    """Test that created class is actually a type object."""
    from template import create_simple_class

    MyClass = create_simple_class('MyClass', {})

    assert isinstance(MyClass, type)
    assert type(MyClass) == type


def test_multiple_instances():
    """Test creating multiple instances."""
    from template import create_simple_class

    Counter = create_simple_class('Counter', {'initial': 0})

    c1 = Counter()
    c2 = Counter()

    assert c1 is not c2
    assert c1.initial == 0
    assert c2.initial == 0


def test_class_with_string_attributes():
    """Test class with string attribute values."""
    from template import create_simple_class

    Config = create_simple_class('Config', {
        'host': 'localhost',
        'port': '8080'
    })

    assert Config.host == 'localhost'
    assert Config.port == '8080'
