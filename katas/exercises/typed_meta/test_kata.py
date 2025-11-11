"""Tests for typed_meta kata."""

import pytest


try:
    from user_kata import TypedMeta
except ImportError:
    from .reference import TypedMeta


def test_correct_types():
    """Test TypedMeta allows correct types."""

    class Person(metaclass=TypedMeta):
        attribute_types = {'name': str, 'age': int}
        name = "Alice"
        age = 30

    p = Person()
    assert p.name == "Alice"
    assert p.age == 30


def test_wrong_type_raises_error():
    """Test TypedMeta raises error for wrong type."""

    with pytest.raises(TypeError, match="must be int, got str"):
        class Person(metaclass=TypedMeta):
            attribute_types = {'age': int}
            age = "thirty"


def test_multiple_wrong_types():
    """Test TypedMeta detects type errors."""

    with pytest.raises(TypeError, match="must be"):
        class Config(metaclass=TypedMeta):
            attribute_types = {'host': str, 'port': int}
            host = "localhost"
            port = "8080"  # Should be int


def test_no_validation():
    """Test TypedMeta works without attribute_types."""

    class Simple(metaclass=TypedMeta):
        value = 42

    s = Simple()
    assert s.value == 42


def test_partial_validation():
    """Test TypedMeta only validates specified attributes."""

    class Mixed(metaclass=TypedMeta):
        attribute_types = {'name': str}
        name = "Alice"
        age = 30  # Not validated
        count = "many"  # Not validated

    m = Mixed()
    assert m.name == "Alice"
    assert m.age == 30


def test_bool_type():
    """Test TypedMeta with bool type."""

    class Flags(metaclass=TypedMeta):
        attribute_types = {'active': bool, 'debug': bool}
        active = True
        debug = False

    f = Flags()
    assert f.active is True
    assert f.debug is False


def test_list_type():
    """Test TypedMeta with list type."""

    class Container(metaclass=TypedMeta):
        attribute_types = {'items': list}
        items = [1, 2, 3]

    c = Container()
    assert c.items == [1, 2, 3]


def test_custom_class_type():
    """Test TypedMeta with custom class type."""

    class Address:
        pass

    class Person(metaclass=TypedMeta):
        attribute_types = {'address': Address}
        address = Address()

    p = Person()
    assert isinstance(p.address, Address)
