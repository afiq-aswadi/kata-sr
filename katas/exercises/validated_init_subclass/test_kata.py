"""Tests for validated_init_subclass kata."""

import pytest


try:
    from user_kata import ValidatedBase
except ImportError:
    from .reference import ValidatedBase


def test_success_with_required_attrs():
    """Test ValidatedBase allows class with required attributes."""

    class User(ValidatedBase, required_attrs=['name', 'email']):
        name = "John"
        email = "john@example.com"

    u = User()
    assert u.name == "John"
    assert u.email == "john@example.com"


def test_missing_attribute_raises_error():
    """Test ValidatedBase raises error for missing attribute."""

    with pytest.raises(TypeError, match="missing required attribute: email"):
        class BadUser(ValidatedBase, required_attrs=['name', 'email']):
            name = "John"
            # email is missing


def test_no_validation():
    """Test ValidatedBase works without required_attrs."""

    class Simple(ValidatedBase):
        value = 42

    s = Simple()
    assert s.value == 42


def test_multiple_required_attrs():
    """Test ValidatedBase validates multiple attributes."""

    class Model(ValidatedBase, required_attrs=['a', 'b', 'c']):
        a = 1
        b = 2
        c = 3

    m = Model()
    assert m.a == 1
    assert m.b == 2
    assert m.c == 3


def test_empty_required_attrs():
    """Test ValidatedBase with empty required_attrs list."""

    class Empty(ValidatedBase, required_attrs=[]):
        value = 100

    e = Empty()
    assert e.value == 100


def test_with_methods():
    """Test ValidatedBase works with methods."""

    class Calculator(ValidatedBase, required_attrs=['name']):
        name = "BasicCalc"

        def add(self, a, b):
            return a + b

    calc = Calculator()
    assert calc.add(2, 3) == 5


def test_none_value():
    """Test required attribute can be None."""

    class Config(ValidatedBase, required_attrs=['setting']):
        setting = None

    c = Config()
    assert c.setting is None


def test_inherited_attributes():
    """Test ValidatedBase accepts inherited attributes."""

    class Base(ValidatedBase):
        name = "base"

    class Child(Base, required_attrs=['name']):
        pass  # name is inherited from Base

    # Should work because name is inherited
    c = Child()
    assert c.name == "base"
