"""Tests for validated_meta kata."""

import pytest


def test_success_with_all_attributes():
    """Test ValidatedMeta allows class with all required attributes."""
    from template import ValidatedMeta

    class User(metaclass=ValidatedMeta):
        required_attributes = ['name', 'email']
        name = "John"
        email = "john@example.com"

    u = User()
    assert u.name == "John"
    assert u.email == "john@example.com"


def test_missing_one_attribute():
    """Test ValidatedMeta raises error for missing attribute."""
    from template import ValidatedMeta

    with pytest.raises(TypeError, match="missing required attribute: email"):
        class InvalidUser(metaclass=ValidatedMeta):
            required_attributes = ['name', 'email']
            name = "John"
            # email is missing


def test_missing_multiple_attributes():
    """Test ValidatedMeta detects first missing attribute."""
    from template import ValidatedMeta

    with pytest.raises(TypeError, match="missing required attribute"):
        class Config(metaclass=ValidatedMeta):
            required_attributes = ['host', 'port', 'database']
            host = "localhost"
            # port and database are missing


def test_no_requirements():
    """Test ValidatedMeta works without required_attributes."""
    from template import ValidatedMeta

    class Product(metaclass=ValidatedMeta):
        price = 100

    p = Product()
    assert p.price == 100


def test_empty_requirements():
    """Test ValidatedMeta works with empty required_attributes."""
    from template import ValidatedMeta

    class Service(metaclass=ValidatedMeta):
        required_attributes = []
        value = 42

    s = Service()
    assert s.value == 42


def test_many_required_attributes():
    """Test ValidatedMeta accepts class with many required attributes."""
    from template import ValidatedMeta

    class Model(metaclass=ValidatedMeta):
        required_attributes = ['a', 'b', 'c', 'd']
        a = 1
        b = 2
        c = 3
        d = 4

    m = Model()
    assert m.a == 1
    assert m.d == 4


def test_with_methods():
    """Test ValidatedMeta works with methods."""
    from template import ValidatedMeta

    class Calculator(metaclass=ValidatedMeta):
        required_attributes = ['name']
        name = "BasicCalc"

        def add(self, a, b):
            return a + b

    calc = Calculator()
    assert calc.add(2, 3) == 5


def test_none_value_accepted():
    """Test ValidatedMeta accepts None as a valid value."""
    from template import ValidatedMeta

    class Config(metaclass=ValidatedMeta):
        required_attributes = ['setting']
        setting = None

    c = Config()
    assert c.setting is None


def test_inherited_attributes():
    """Test ValidatedMeta accepts inherited attributes."""
    from template import ValidatedMeta

    class Base(metaclass=ValidatedMeta):
        name = "base"

    class Child(Base):
        required_attributes = ['name']
        # name is inherited from Base

    # Should work because name is inherited
    c = Child()
    assert c.name == "base"
